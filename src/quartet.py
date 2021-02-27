import time

import numpy as np
import torch
import random
from pointcloud import PointCloud
import itertools
from scipy.linalg import null_space
from mesh_to_sdf import mesh_to_sdf

from tetrahedral_group import TetsGroupSharesVertex
import os


def tensors_eq(v1, v2):
    return len(v1) == len(v2) and (v1 == v2).sum() == len(v2)


class Tetrahedron:
    def __init__(self, vertices, tet_num, depth=0):
        self.vertices = sorted(vertices)
        self.occupancy = torch.tensor(1. / 6000)  # torch.rand(1)  # very small chance to all be 0
        self.neighborhood = set()

        self.tet_num = torch.tensor(tet_num, dtype=torch.long)
        self.features = torch.tensor([0.])

        # self.features = torch.cat([v.loc for v in self.vertices])
        # self.features = torch.cat([self.features, self.center().loc])

        # self.features = torch.stack([v.loc for v in self.vertices]).permute(1, 0).sum(dim=-1) / 4.
        # self.features = torch.rand(30)
        # rand_vec = torch.rand(3) - 1
        # self.features = torch.stack([v.loc for v in self.vertices]).permute(1, 0).sum(dim=-1) / 4. + rand_vec
        # self.features = torch.cat([self.features, torch.rand(27, requires_grad=True)])
        self.prev_features = self.features
        self.sub_divided = None
        self.pooled = False
        self.depth = depth

        self.half_faces = []
        self.faces_by_vertex = {}
        self.faces_by_vertex_opposite = {}

        self.init_features = None
        self.init_vertices = None
        self.init_occupancy = None

        self.set_as_init_values()

    def set_as_init_values(self):
        self.init_features = self.features.clone()
        self.init_occupancy = self.occupancy.clone()

    def sample_points(self, n):
        a, b, c, d = [v.curr_loc for v in self.vertices]
        t1 = b - a
        t2 = c - a
        t3 = d - a

        q_lst = []

        while len(q_lst) < n:
            # q = np.random.rand(3)
            # if np.sum(q) <= 1:
            #     q_lst.append(q)
            q = np.random.rand(4)
            q[1:] /= np.sum(q[1:])
            q *= q[0]
            q_lst.append(q[1:])

        return [a + q[0] * t1 + q[1] * t2 + q[2] * t3 for q in q_lst]

    def add_neighbor(self, neighbor):
        self.neighborhood.add(neighbor)

    def remove_neighbor(self, neighbor):
        self.neighborhood.remove(neighbor)

    def update_occupancy(self, new_occupancy):
        self.occupancy = new_occupancy

    def is_neighbor(self, other):
        c = 0
        for v1 in self.vertices:
            for v2 in other.vertices:
                if tensors_eq(v1.original_loc, v2.original_loc):
                    c += 1
        return c == 3

    def center(self):
        a = torch.stack([v.curr_loc for v in self.vertices])
        loc = a.permute(1, 0).sum(dim=1) / 4.
        loc = loc.type(torch.float)
        return Vertex(*loc)

    def __hash__(self):
        return (self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]).__hash__()

    def __iter__(self):
        return iter(self.vertices)

    def update_by_deltas(self, vertices_deltas):
        for m, v in zip(vertices_deltas, self.vertices):
            v.update_vertex(m)

    def update_move_signed_distance(self, vertices_deltas):
        for m, v in zip(vertices_deltas, self.vertices):
            v.update_sd_loss(m)

    @staticmethod
    def determinant_3x3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]) +
                m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]))

    @staticmethod
    def subtract(a, b):
        return (a[0] - b[0],
                a[1] - b[1],
                a[2] - b[2])

    def volume(self):
        p1, p2, p3, p4 = [v.curr_loc for v in self.vertices]
        vol = torch.det(torch.stack([p1 - p4, p2 - p4, p3 - p4])) / 6
        return vol.abs()

    def translate(self, vec):
        for vert in self.vertices:
            vert.update_vertex(vec)

    def reset(self):
        self.features = self.init_features.clone().to(self.features.device)
        self.occupancy = self.init_occupancy.clone()
        for v in self.vertices:
            v.reset()

    def get_faces(self):
        return list(itertools.combinations(self.vertices, 3))

    def calculate_half_faces(self, force=False):
        if force:
            self.half_faces = []
        assert len(self.half_faces) == 0
        for nei in self.neighborhood:
            if self.is_neighbor(nei):  # can also be on the boundary so we add itself as a neighbor
                face_coords = [v.original_loc for v in intersect(self, nei)]
                self.half_faces.append(HalfFace(face_coords, (self, nei)))

        new_coords = []
        for face_coords in list(itertools.combinations(self.vertices, 3)):
            is_new = True
            for hf in self.half_faces:
                if hf.is_eq(face_coords):
                    is_new = False
            if is_new:
                new_coords.append(face_coords)
        for coords in new_coords:
            self.half_faces.append(HalfFace([v.original_loc for v in coords], (self, self)))

        ##########################################################################
        for v in self.vertices:
            self.faces_by_vertex[v.get_original_xyz()] = [hf for hf in self.half_faces if hf.has(v.original_loc)]
            for hf in self.faces_by_vertex[v.get_original_xyz()]:
                assert abs(hf.plane.signed_distance(v.original_loc)) < 0.01
            tmp = [hf for hf in self.half_faces if not hf.has(v.original_loc)]
            assert len(tmp) == 1
            self.faces_by_vertex_opposite[v.get_original_xyz()] = tmp[0]

        self.half_faces[0].set_orientation()

        # tests:
        center = self.center().original_loc
        sides = [hf.plane.get_point_side(center) for hf in self.half_faces]
        for b in sides:
            assert sides[0] == b

    def get_half_faces(self):
        return self.half_faces

    # def subdivide(self):
    #     """
    #      subdivide tet into 4 tets that their shared vertex is the center
    #     """
    #
    #     ret_tets = []
    #     center = self.center()
    #     unique_neighbors = []
    #
    #     vertex_to_tets_involved = {}
    #     for i, half_face in enumerate(self.half_faces):
    #         vertices_i = half_face.get_vertices() + [center]
    #         ret_tets.append(Tetrahedron(vertices_i, tet_num=self.tet_num.item()))
    #
    #         unique_nei = half_face.tets[1]
    #         unique_neighbors.append(unique_nei)
    #         unique_nei.remove_neighbor(self)
    #         unique_nei.add_neighbor(ret_tets[i])
    #
    #         for v in half_face.coords:
    #             if v not in vertex_to_tets_involved:
    #                 vertex_to_tets_involved[tuple(v.numpy())] = []
    #             vertex_to_tets_involved[tuple(v.numpy())].append(ret_tets[i])
    #
    #     for i in range(4):
    #         neighbors = [ret_tets[j] for j in range(4) if j != i] + [
    #             unique_neighbors[i] if unique_neighbors[i] != self else ret_tets[i]]
    #         ret_tets[i].neighborhood = set(neighbors)
    #     for i in range(4):
    #         ret_tets[i].calculate_half_faces(force=True)
    #
    #     center.set_tets_group(ret_tets)
    #     for v in self.vertices:
    #         v.tets_group.update_tets_list([self], vertex_to_tets_involved[v.get_original_xyz()])  # TODO
    #
    #     return ret_tets, center

    def subdivide(self):
        """
         subdivide tet into 4 tets that their shared vertex is the center
        """

        ret_tets = []
        center = self.center()
        for i, half_face in enumerate(self.half_faces):
            vertices_i = half_face.get_vertices() + [center]
            ret_tets.append(Tetrahedron(vertices_i, tet_num=self.tet_num.item()))

        return ret_tets, center


def vertices_intersection(ver1, ver2):
    intersection = []
    for v1 in ver1:
        exist = False
        for v2 in ver2:
            if tensors_eq(v1.original_loc, v2.original_loc):
                exist = True
        if exist:
            intersection.append(v1)
    return intersection


def intersect(tet1, tet2):  # TODO: change name
    return vertices_intersection(tet1.vertices, tet2.vertices)


class Plane:

    def __init__(self, a, b, c, d):
        Z = (a ** 2 + b ** 2 + c ** 2) ** 0.5
        self.a = a / Z
        self.b = b / Z
        self.c = c / Z
        self.d = d / Z

    def change_orientation(self):
        self.a = -self.a
        self.b = -self.b
        self.c = -self.c
        self.d = -self.d

    def get_normal(self):
        return torch.tensor([self.a, self.b, self.c], dtype=torch.float)

    def get_point_side(self, x):
        """ True is right, False is left"""
        return self.a * x[0] + self.b * x[1] + self.c * x[2] + self.d <= 0.  # negative inside positive outside

    def signed_distance(self, x):
        return self.a * x[0] + self.b * x[1] + self.c * x[2] + self.d


class HalfFace:
    def __init__(self, coords, tets):
        self.coords = coords

        # calculate plane according to 3 points
        mat = np.array([c.numpy() for c in coords])
        if np.linalg.matrix_rank(mat) == 3:
            b = -np.ones(3)
            solution = np.linalg.solve(mat, b)
            self.plane = Plane(solution[0], solution[1], solution[2], 1.)
        else:
            mat_null_space = null_space(mat)
            # solution = mat_null_space[:, 0]
            if mat_null_space.shape[1] > 0:
                solution = mat_null_space[:, 0]
            else:
                eigs = np.linalg.eig(mat)
                solution = eigs[1][:, np.argmin(abs(eigs[0]))]

            self.plane = Plane(solution[0], solution[1], solution[2], 0.)

        self.tets = tets
        self.oriented = False  # all half faces of the same tet are with same orientation

    def set_orientation(self):
        if self.oriented:
            return None
        tet_half_faces = self.tets[0].get_half_faces()
        center = self.tets[0].center().original_loc
        for half_face in tet_half_faces:
            if not half_face.plane.get_point_side(center):
                half_face.plane.change_orientation()

        center = self.tets[0].center().original_loc
        sides = [hf.plane.get_point_side(center) for hf in tet_half_faces]
        assert not ((not sides[0]) in sides)

        self.oriented = True

    def get_vertices(self):
        return [Vertex(*c.numpy()) for c in self.coords]

    def is_eq(self, three_vertices):
        return len(vertices_intersection(three_vertices, [Vertex(*x) for x in self.coords])) == 3

    def get_neighbor(self):
        return self.tets[1]

    def has(self, v):
        for v2 in self.coords:
            if tensors_eq(v, v2):
                return True

        return False


class Vertex:
    def __init__(self, x, y, z):
        self.curr_loc = torch.tensor([x, y, z], dtype=torch.float32)
        self.original_loc = self.curr_loc.detach().clone()
        self.on_boundary = self.is_on_boundary()

        self.tets_group = None
        self.last_update_signed_distance = None

    def is_on_boundary(self):
        for i in range(3):
            if self.curr_loc[i] > 0.99 or self.curr_loc[i] < 0.01:
                return True
        return False

    def update_sd_loss(self, move_vector):
        if not self.on_boundary:
            a, b = self.tets_group.query_direction(move_vector)  # TODO: a,b...
            self.last_update_signed_distance[0].append(a)
            self.last_update_signed_distance[1].append(b)

    def update_vertex(self, move_vector):
        if not self.on_boundary:
            self.curr_loc = self.curr_loc + move_vector

    def reset(self):
        self.curr_loc = self.original_loc.detach().clone()
        self.last_update_signed_distance = None

    def get_curr_xyz(self):
        x, y, z = self.curr_loc[0].item(), self.curr_loc[1].item(), self.curr_loc[2].item()
        return x, y, z

    def get_original_xyz(self):
        x, y, z = self.original_loc[0].item(), self.original_loc[1].item(), self.original_loc[2].item()
        return x, y, z

    def __hash__(self):
        x, y, z = self.get_original_xyz()
        return (x, y, z).__hash__()

    def __eq__(self, other):
        return self.get_original_xyz() == other.get_original_xyz()

    def __ge__(self, other):
        x, y, z = self.original_loc
        ox, oy, oz = other.original_loc
        if x < ox:
            return False
        if x > ox:
            return True

        if y < oy:
            return False
        if y > oy:
            return True

        if z < oz:
            return False
        return True

    def __lt__(self, other):
        return not (self >= other)

    def set_tets_group(self, tets_list):
        assert self.tets_group is None
        self.tets_group = TetsGroupSharesVertex(self, tets_list)


class QuarTet:
    def __init__(self, path='../cube_0.05.tet', device='cpu', metadata_path=None):
        self.curr_tetrahedrons = None
        self.vertices = None
        self.load(path, device, meta_data_path=metadata_path)

        self.last_vertex_update_average = None
        self.device = device

    def calculate_and_update_neighborhood(self):
        vertices_to_tets_dict = dict()
        for vertex in self.vertices:
            vertices_to_tets_dict[vertex] = set()
        for tet in self.curr_tetrahedrons:
            for vertex in tet:
                vertices_to_tets_dict[vertex].add(tet)

        for tet in self.curr_tetrahedrons:
            for face in tet.get_faces():
                neighbor_set = set.intersection(*[vertices_to_tets_dict[vertex] for vertex in face])
                assert (len(neighbor_set) == 1 or len(neighbor_set) == 2)
                for neighbor in neighbor_set:
                    if neighbor != tet:
                        neighbor.add_neighbor(tet)
                        tet.add_neighbor(neighbor)

    def fill_neighbors(self):
        for tet in self.curr_tetrahedrons:
            for i in range(4 - len(tet.neighborhood)):
                tet.add_neighbor(tet)

    def merge_same_vertices(self):
        all_vertices = {}
        for tet in self.curr_tetrahedrons:
            for v in tet.vertices:
                all_vertices[v.get_original_xyz()] = v

        for tet in self.curr_tetrahedrons:
            new_vertices = []
            for v in tet.vertices:
                new_vertices.append(all_vertices[v.get_original_xyz()])
            tet.vertices = new_vertices

    def zero_grad(self):
        for tet in self.curr_tetrahedrons:
            tet.features = tet.features.detach().clone()
            tet.prev_features = tet.prev_features.detach().clone()

    # def sample_disjoint_faces(self, N):  # TODO: do it exact N
    #     faces = []
    #     visited_tets = set()
    #     sampled_indices = np.random.randint(0, len(self.curr_tetrahedrons), size=N)
    #     for idx in sampled_indices:
    #         tet = self.curr_tetrahedrons[idx]
    #         if tet in visited_tets:
    #             continue
    #
    #         neighbor_idx = np.random.randint(0, len(tet.neighborhood))
    #         neighbor = list(tet.neighborhood)[neighbor_idx]
    #         if neighbor != tet:
    #             visited_tets.add(tet)
    #             visited_tets.add(neighbor)
    #             faces.append(Face(tet, neighbor))
    #
    #     return faces

    def __iter__(self):
        return self.curr_tetrahedrons.__iter__()

    def __len__(self):
        return len(self.curr_tetrahedrons)

    def get_centers(self):
        centers = []
        for tet in self.curr_tetrahedrons:
            centers.append(tet.center().original_loc)
        return torch.stack(centers)

    def get_occupied_centers(self):
        occupied_centers = []
        for tet in self.curr_tetrahedrons:
            if tet.occupancy >= 0.5:
                occupied_centers.append(tet.center().original_loc)
        return torch.stack(occupied_centers)

    def update_occupancy_using_sdf(self, sdf):
        for i, tet in enumerate(self.curr_tetrahedrons):
            if sdf[i] <= 0:
                tet.occupancy = torch.tensor(1.)
            else:
                tet.occupancy = torch.tensor(0.)

    def update_occupancy_using_convex_hull(self, convex_hull_mesh):
        signs = mesh_to_sdf(convex_hull_mesh, np.array(self.get_centers()))
        self.update_occupancy_using_sdf(signs)

    def sample_point_cloud(self, pc_size):
        samples_weights = []
        for tet in self.curr_tetrahedrons:
            samples_weights.append((tet.center().curr_loc, tet.occupancy))  # grad of 1
        samples = torch.stack([x[0] for x in samples_weights])
        weights = torch.stack([x[1] for x in samples_weights])
        return samples, weights

    # def sample_point_cloud(self, pc_size):
    #
    #     samples_weights = []
    #     for tet in self.curr_tetrahedrons:
    #         samples_weights.append((tet.center().loc, tet.occupancy))  # grad of 1
    #     samples_weights = random.choices(samples_weights, k=pc_size)
    #     samples = torch.stack([x[0] for x in samples_weights])
    #     weights = torch.stack([x[1] for x in samples_weights])
    #     return samples, weights
    #
    # def sample_point_cloud_2(self, pc_size):
    #     occupied_tets = self.curr_tetrahedrons
    #     volumes = [tet.volume() * tet.occupancy for tet in occupied_tets]
    #     volumes_total = sum(volumes)
    #
    #     points_count = [np.int(np.ceil(((volume / volumes_total) * pc_size).item())) for volume in volumes]
    #
    #     samples = []
    #     for i, tet in enumerate(occupied_tets):
    #         samples.extend(tet.sample_points(points_count[i]))
    #
    #     samples = random.choices(samples, k=pc_size)
    #     return torch.stack(samples)

    def sample_point_cloud(self, pc_size):
        occupied_tets = self.curr_tetrahedrons
        volumes = [tet.volume().abs() * tet.occupancy for tet in occupied_tets]
        # volumes = [tet.occupancy for tet in occupied_tets]
        volumes_total = sum(volumes)

        points_count = [np.int(np.ceil(((volume / volumes_total) * pc_size).item())) for volume in volumes]
        # points_count = [5 for volume in volumes]

        samples = []
        for i, tet in enumerate(occupied_tets):
            # if points_count[i] <= 1:
            #     continue
            for _ in range(points_count[i]):
                r = np.random.rand(4)
                r /= np.sum(r)
                samples.append(sum([r[i] * tet.vertices[i].curr_loc for i in range(4)]))

        # if len(samples) == 0:
        #     return torch.rand(pc_size, 3)
        samples = random.choices(samples, k=pc_size)
        # return torch.stack(samples), volumes
        return torch.stack(samples)

    def sample_point_cloud_2(self, pc_size):

        samples_weights = []
        for tet in self.curr_tetrahedrons:
            samples_weights.append((tet.center().original_loc, tet.occupancy))  # grad of 1
        samples_weights = random.choices(samples_weights, k=pc_size)
        samples = torch.stack([x[0] for x in samples_weights])
        weights = torch.stack([x[1] for x in samples_weights])
        return samples, weights

    # def sample_point_cloud_4(self, pc_size):
    #     occupied_tets = self.curr_tetrahedrons
    #     # volumes = [tet.volume() * tet.occupancy for tet in occupied_tets]
    #     volumes = [(tet.occupancy > 0.5) + 0.01 for tet in occupied_tets]
    #     volumes_total = sum(volumes)
    #
    #     points_count = [np.int(np.ceil(((volume / volumes_total) * pc_size).item())) for volume in volumes]
    #
    #     samples = []
    #     for i, tet in enumerate(occupied_tets):
    #         little_qube = (torch.rand(points_count[i], 3) - 0.5) / 50
    #         little_qube = little_qube + tet.center().loc.expand_as(little_qube)
    #         samples.append(little_qube)
    #     return torch.cat(samples)

    def export_metadata(self, path):
        occupancies_str = []
        neighborhoods_str = []
        for tet in self.curr_tetrahedrons:
            occupancies_str.append(f"{tet.occupancy.item()}\n")
            neighborhoods_str.append(', '.join([str(neighbor.tet_num.item()) for neighbor in tet.neighborhood]))

        with open(path, 'w') as output_file:
            output_file.write("occupancies:\n")
            output_file.write(''.join(occupancies_str))
            output_file.write("neighborhoods:\n")
            output_file.write('\n'.join(neighborhoods_str))

    def export_tet(self, path, export_only_filled=False):
        vertex_to_idx = {}
        n_ver = 0
        to_write = []
        for tet in self.curr_tetrahedrons:
            if not export_only_filled or tet.occupancy > 0.5:
                for v in tet.vertices:
                    if v not in vertex_to_idx:
                        x, y, z = v.curr_loc
                        to_write.append(f"{x} {y} {z}\n")
                        vertex_to_idx[v] = n_ver
                        n_ver += 1
        for tet in self.curr_tetrahedrons:
            if not export_only_filled or tet.occupancy > 0.5:
                indices = [vertex_to_idx[v] for v in tet.vertices]
                to_write.append(f"{indices[0]} {indices[1]} {indices[2]} {indices[3]}\n")
        with open(path, "w") as output_file:
            output_file.write(f"tet {n_ver} {len(self.curr_tetrahedrons)}\n")
            output_file.write(''.join(to_write))


    def export(self, path, export_metadata=True):
        # exporting all the tetrahedrons
        self.export_tet(path)

        name_without_extension, extension = os.path.splitext(path)
        # exporting the metadata
        if export_metadata:
            self.export_metadata(f'{name_without_extension}_data.data')

        # exporting only the occupied tetrahedrons
        self.export_tet(f'{name_without_extension}_filled{extension}', export_only_filled=True)

    def load_tetrahedrons(self, path):
        self.curr_tetrahedrons = []
        vertices = []
        with open(path, "r") as input_file:
            first_line = input_file.readline()
            first_line_split = first_line.split(' ')
            print(f'Loading tet file\nReading {int(first_line_split[1])} vertices')
            for i in range(int(first_line_split[1])):
                line = input_file.readline()
                coordinates = line.split(' ')
                vertices.append(Vertex(*[round(float(c), ndigits=5) for c in coordinates]))
                if i % 2000 == 0 and i > 0:
                    print(f'Read {i} vertices')

            self.vertices = vertices
            print(f'Finished reading vertices\nReading {int(first_line_split[2])} tetrahedrons')
            for i in range(int(first_line_split[2])):
                line = input_file.readline()
                if line == '':
                    continue
                vertex_indices = line.split(' ')
                self.curr_tetrahedrons.append(
                    Tetrahedron([vertices[int(i)] for i in vertex_indices], len(self.curr_tetrahedrons)))
                if i % 2000 == 0 and i > 0:
                    print(f'Read {i} tetrhedrons')


    def load_meta_data(self, path, device='cpu'):
        with open(path, 'r') as meta_data_input:
            if next(meta_data_input).strip() != "occupancies:":
                raise IOError("Wrong file content format")
            try:
                quartet_iter = iter(self)
                next_line = next(meta_data_input).strip()
                while next_line != "neighborhoods:":
                    tet = next(quartet_iter)
                    tet.occupancy = torch.tensor(float(next_line), device=device)
                    next_line = next(meta_data_input).strip()

                next_line = next(meta_data_input).strip()
                for tet in self:
                    curr_neighborhood = []
                    neighbor_indexes = next_line.split(', ')
                    for idx in neighbor_indexes:
                        curr_neighborhood.append(self.curr_tetrahedrons[int(idx)])
                    tet.neighborhood = curr_neighborhood.copy()
                    next_line = next(meta_data_input).strip()

            except StopIteration:
                pass


    def load(self, path, device, meta_data_path=None):
        """
        Loads a quartet from a file
        :param path: the path to the quartet file in .tet format
        :param device: the device to move the quartet to (e.g. 'cpu', 'cuda')
        :param meta_data_path: an optional parameter representing a path to a file containing the metadata of the quartet
                object, containing occupancies values of each tetrahedron and the neighborhoods of each tetrahedron.
                if 'default', the method uses the default name used the the quartet while saving, if None no file is
                loaded and initialization is done using default tetrahedron occupancies initialization and manual
                neighborhoods calculaction
        :return: None
        """
        self.load_tetrahedrons(path)

        if meta_data_path is not None:
            print("Loading metadata")
            if meta_data_path == 'default':
                base_name = os.path.splitext(path)[0]
                meta_data_path = f"{base_name}_data.data"

            if os.path.exists(meta_data_path):
                self.load_meta_data(meta_data_path, device)
        else:
            self.do_init_calculations()

        for tet in self.curr_tetrahedrons:
            tet.occupancy = tet.occupancy.cpu()
            tet.features = tet.features.to(device)
            tet.tet_num = tet.tet_num.to(device)
            tet.prev_features = tet.prev_features.to(device)
            for i in range(4):
                tet.vertices[i].curr_loc = tet.vertices[i].curr_loc.cpu()

        self.init_setup()

    def do_init_calculations(self):
        print('Calculating neighborhoods')
        self.calculate_and_update_neighborhood()
        print('Filling neighbors')
        self.fill_neighbors()
        print('Merge same vertices')
        self.merge_same_vertices()

    def init_setup(self):
        for tet in self.curr_tetrahedrons:
            tet.features.requires_grad_()

        for tet in self.curr_tetrahedrons:
            tet.calculate_half_faces()

        for tet in self.curr_tetrahedrons:
            tet.half_faces[0].set_orientation()

        vertex_tet_dict = {}
        for tet in self.curr_tetrahedrons:
            for v in tet.vertices:
                if v not in vertex_tet_dict:
                    vertex_tet_dict[v] = []
                vertex_tet_dict[v].append(tet)

        for v, v_tets_list in vertex_tet_dict.items():
            v.set_tets_group(v_tets_list)
            for tet in v_tets_list:
                assert len(tet.faces_by_vertex) != 0

        for tet in self.curr_tetrahedrons:
            tet.set_as_init_values()

    def reset(self):
        for tet in self.curr_tetrahedrons:
            tet.reset()

    def create_mesh(self):
        faces = set()
        for tet in self.curr_tetrahedrons:
            for nei in tet.neighborhood:
                if tet == nei:
                    continue
                if (tet.occupancy > 0.5) ^ (nei.occupancy > 0.5):
                    face_container = tuple(intersect(tet, nei))
                    if face_container not in faces:
                        faces.add(face_container)

                # checking if the face is on the boundary and the tetrahedron is occupied
                # and therefore is part of the mesh
                if tet.occupancy > 0.5:
                    for face in tet.get_faces():
                        face_container = tuple(face)
                        if face_container not in faces and all([v.on_boundary for v in face_container]):
                            faces.add(face_container)
        return faces

    def export_mesh(self, path):
        faces = self.create_mesh()
        obj_file_str_vert = []
        obj_file_str_faces = []
        vertices = {}
        c = 1
        for i, f_coords in enumerate(faces):
            for v in f_coords:
                if v not in vertices:
                    vertices[v] = c
                    c += 1
                    x, y, z = v.curr_loc
                    obj_file_str_vert.append(f"v {x} {y} {z}")
            obj_file_str_faces.append(
                f"f {vertices[f_coords[0]]} {vertices[f_coords[1]]} {vertices[f_coords[2]]}")
        with open(path, 'w+') as f:
            f.write("\n".join(obj_file_str_vert))
            f.write("\n")
            f.write("\n".join(obj_file_str_faces))

    def export_point_cloud(self, path, n=2500):
        # points, _ = self.sample_point_cloud_2(N)
        s = time.time()
        print("Start sampling pts")
        points = self.sample_point_cloud(n)
        print(f"Done {time.time() - s}")
        pc = PointCloud()
        pc.init_with_points(points)
        pc.write_to_file(path)

    def fill_sphere(self):
        cube_center = torch.tensor([[0.5, 0.5, 0.5]])
        for tet in self.curr_tetrahedrons:
            if torch.cdist(tet.center().curr_loc.unsqueeze(0), cube_center) <= 0.5:
                tet.occupancy = torch.tensor(1.)
                tet.init_occupancy = tet.occupancy.clone()
            else:
                tet.occupancy = torch.tensor(0.)
                tet.init_occupancy = tet.occupancy.clone()

    # def subdivide_tets(self, net):
    #     new_tetrahedrons = []
    #     for i, tet in enumerate(self.curr_tetrahedrons):
    #         new_tets, center = tet.subdivide()
    #         self.vertices.append(center)
    #         new_tetrahedrons.extend(new_tets)
    #         for tet2 in new_tets:
    #             assert tet not in tet2.neighborhood
    #     for tet1 in self.curr_tetrahedrons:
    #         if not (True in [v.on_boundary for v in tet1.vertices]):
    #             for tet2 in new_tetrahedrons:
    #                 assert tet1 not in tet2.neighborhood
    #     # TODO:
    #     # set tets numbers
    #     # set new tet embedding based on the previous
    #
    #     old_embedding = net.tet_embed.state_dict()["weight"]
    #     new_embedding_weights = torch.empty((old_embedding.shape[0] * 4, *old_embedding.shape[1:]))
    #
    #     for i, tet in enumerate(new_tetrahedrons):
    #         new_embedding_weights[i] = old_embedding[tet.tet_num]
    #         tet.tet_num = torch.tensor(i)
    #
    #     new_embedding_weights.requires_grad_()
    #     net.tet_embed = torch.nn.Embedding(*new_embedding_weights.shape)
    #     net.tet_embed.load_state_dict({"weight": new_embedding_weights})
    #
    #     self.curr_tetrahedrons = new_tetrahedrons

    def subdivide_tets(self, net):
        new_tetrahedrons = []
        vertices = []
        for i, tet in enumerate(self.curr_tetrahedrons):
            new_tets, center = tet.subdivide()
            new_vertices = [Vertex(*v.get_original_xyz()) for v in tet.vertices] + [center]
            vertices.extend(new_vertices)
            new_tetrahedrons.extend(new_tets)
        self.vertices = vertices
        self.curr_tetrahedrons = new_tetrahedrons

        # neighbors + half faces calculations
        self.do_init_calculations()

        # update embeddings
        old_embedding = net.tet_embed.state_dict()["weight"]
        new_embedding_weights = torch.empty((old_embedding.shape[0] * 4, *old_embedding.shape[1:]))

        for i, tet in enumerate(new_tetrahedrons):
            new_embedding_weights[i] = old_embedding[tet.tet_num]
            tet.tet_num = torch.tensor(i, device=old_embedding.device)
        new_embedding_weights.to(old_embedding.device)
        new_embedding_weights.requires_grad_()
        net.tet_embed = torch.nn.Embedding(*new_embedding_weights.shape)
        net.tet_embed.load_state_dict({"weight": new_embedding_weights})
        net.tet_embed.to(old_embedding.device)

        self.curr_tetrahedrons = new_tetrahedrons

    def fix_at_position(self):
        """
        can be applied only if vertices_movement_bound_loss == 0.
        """
        for v in self.vertices:
            v.original_loc = v.curr_loc.detach().clone()
            # TODO:
            # update v.tets_group (just set a new center) - not sure needed

        for tet in self.curr_tetrahedrons:
            tet.calculate_half_faces(force=True)



    def fill_occupancy_with_sdf(self):
        """ leave it to nitzan"""


    def __getitem__(self, index):
        return self.curr_tetrahedrons[index]


if __name__ == '__main_1_':
    # a = QuarTet(2, 'cpu')
    a = QuarTet('../objects/cube_0.05.tet')
    a.fill_sphere()
    # for tet in a:
    #     tet.occupancy = torch.tensor(0.)
    # a[5].occupancy = torch.tensor(1.)
    # a.export_point_cloud('./pc.obj', 10000)
    a.export_mesh('./mesh.obj')
    a.export(path='quartet.tet')

    # pc = PointCloud()
    # pc.load_file('./filled_sphere.obj')

if __name__ == '__main_1_':
    a = QuarTet('../objects/cube_0.05.tet')
    a.fill_sphere()
    a.export_mesh('./mesh.obj')
    a.export_point_cloud('./pc.obj', 10000)
    print(a.vertices)

if __name__ == '__main__':
    import networks

    a = QuarTet('../objects/cube_0.15.tet', device='cuda')
    net = networks.OurNet(None, 402).cuda()

    s = time.time()
    a.subdivide_tets(net)
    print(f"Time to subdivide {time.time() - s}")
    s = time.time()
    net(a)
    print(f"net applied {time.time() - s}")
    s = time.time()
    a.fix_at_position()
    print(f"position fixed {time.time() - s}")
    print(len(a.curr_tetrahedrons))
# if __name__ == '__main__':
#     a = QuarTet('../checkpoints/default_name/quartet_100.tet')
#     a.export_mesh('./mesh.obj')
#     a.export_point_cloud('./pc.obj', 10000)
#     print(a.vertices)

