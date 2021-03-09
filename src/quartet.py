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
from sklearn.neighbors import radius_neighbors_graph


def tensors_eq(v1, v2):
    return len(v1) == len(v2) and (v1 == v2).sum() == len(v2)


class Tetrahedron:
    def __init__(self, vertices, tet_num, depth=0):
        self.vertices = sorted(vertices)
        self.occupancy = torch.tensor(1. / 6000)  # torch.rand(1)  # very small chance to all be 0
        self.neighborhood = set()

        self.tet_num = torch.tensor(tet_num, dtype=torch.long)
        self.features = torch.tensor([0.])
        self.prev_features = self.features
        self.sub_divided = None
        self.pooled = False
        self.depth = depth

        self.half_faces = []
        self.faces_by_vertex = {}
        self.faces_by_vertex_opposite = {}

        for v in self.vertices:
            self.faces_by_vertex[v.get_original_xyz()] = []

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
        """ returns the four faces making this tetrahedron as a list of 4 tuples of size 3 of Vertex objects """
        return list(itertools.combinations(self.vertices, 3))

    def add_face(self, neighbor, face):
        """
        The function calculates the half-face corresponding to the new face, and updates the faces_by_vertex
        and faces_by_vertex_opposite dictionaries
        :param neighbor: The neighbor tetrahedron on the other side of the new face, if exists. If set to None the face
            is treated as a face on the edge of the quartet, and therefore we create the half face with the tetrahedron
            as it's own neighbor
        :param face: The new face to add, as an iterable of the three vertices representing the face
        """
        face_coords = [v.original_loc for v in face]
        if neighbor is not None:
            new_half_face = HalfFace(face_coords, (self, neighbor))
        else:
            new_half_face = HalfFace(face_coords, (self, self))
        self.half_faces.append(new_half_face)

        for v in face:
            self.faces_by_vertex[v.get_original_xyz()].append(new_half_face)
        for v in self.vertices:
            if v not in face:
                self.faces_by_vertex_opposite[v.get_original_xyz()] = new_half_face

    def calculate_half_faces(self, force=False):
        """ :param force: If True the half faces will be recalculated """
        if force:
            self.half_faces = []
        assert len(self.half_faces) == 0
        added_faces = set()
        for nei in self.neighborhood:
            if self is not nei:  # can also be on the boundary so we add itself as a neighbor
                face = tets_intersection(self, nei)
                added_faces.add(face)
                self.add_face(nei, face)

        # adding the faces that are on the border of the quartet
        # we treat those faces differently because they don't have a neighbor tetrahedron, and therefore we need
        # another way to calculate the face, without using intersection as we used before
        new_faces = []
        for face in self.get_faces():
            if face not in added_faces:
                new_faces.append(face)
        for face in new_faces:
            self.add_face(None, face)

    def get_half_faces(self):
        return self.half_faces

    def subdivide(self):
        """ subdivide tet into 4 tets that their shared vertex is the center """
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


def tets_intersection(tet1, tet2):  # TODO: change name
    return tuple(vertices_intersection(tet1.vertices, tet2.vertices))


class Plane:
    def __init__(self, a, b, c, d):
        Z = (a ** 2 + b ** 2 + c ** 2) ** 0.5
        self.coefficients = torch.tensor([a, b, c, d], dtype=torch.float) / Z

    def change_orientation(self):
        self.coefficients = -1 * self.coefficients

    def get_normal(self):
        return self.coefficients[:-1].clone()

    def get_point_side(self, x):
        """ True is inside False is outside """
        return self.signed_distance(x) <= 0.  # negative inside positive outside

    def signed_distance(self, x):
        # return self.a * x[0] + self.b * x[1] + self.c * x[2] + self.d
        return (self.coefficients[:-1] * x).sum() + self.coefficients[-1]


class HalfFace:
    """
     A utility class used to describe a face of a tetrahedron with a plane and a normal.
    The plane is saved as four variables representing the plane equation, and a normal vector.
    The class is called HalfFace because we use 2 of these to describe a single face,
    because we want to have two different normals for the same face.
    """

    def __init__(self, coords, tets):
        """
        :param coords: the three points describing the face
        :param tets: the two tetrahedrons that share the face
        """
        self.coords = coords

        # calculating the plane according to 3 points
        mat = np.array([c.numpy() for c in coords])
        if np.linalg.matrix_rank(mat) == 3:
            b = -np.ones(3)
            solution = np.linalg.solve(mat, b)
            self.plane = Plane(solution[0], solution[1], solution[2], 1.)
        else:
            mat_null_space = null_space(mat)
            if mat_null_space.shape[1] > 0:
                solution = mat_null_space[:, 0]
            else:
                eigs = np.linalg.eig(mat)
                solution = eigs[1][:, np.argmin(abs(eigs[0]))]

            self.plane = Plane(solution[0], solution[1], solution[2], 0.)

        self.tets = tets
        self.oriented = False  # all half faces of the same tet are with same orientation
        self.set_orientation()

    def set_orientation(self):
        """
        setting self's plane orientation such that the signed distance to the tetrahedron's center will be positive
        """
        if self.oriented:
            return
        center = self.tets[0].center().original_loc
        result = self.plane.get_point_side(center)
        if not result:
            self.plane.change_orientation()
        self.oriented = True

    def get_vertices(self):
        return [Vertex(*c.numpy()) for c in self.coords]

    def is_eq(self, three_vertices):
        return len(vertices_intersection(three_vertices, self.get_vertices())) == 3

    def get_neighbor(self):
        return self.tets[1]

    def has_vertex(self, v):
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
        self.last_update_signed_distance = []

    def is_on_boundary(self):
        for i in range(3):
            if self.curr_loc[i] > 0.99 or self.curr_loc[i] < 0.01:
                return True
        return False

    def update_sd_loss(self, move_vector):
        if not self.on_boundary:
            # original_distance = self.tets_group.signed_distance_to_face_in_direction(move_vector)
            original_distance = self.tets_group.inner_sphere_loss(move_vector)
            self.last_update_signed_distance.append(original_distance)

    def update_vertex(self, move_vector):
        if not self.on_boundary:
            self.curr_loc = self.curr_loc + move_vector

    def reset(self):
        self.curr_loc = self.original_loc.detach().clone()
        self.last_update_signed_distance = []

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

    def update_occupancy_using_convex_hull(self, convex_hull_mesh):
        signs = mesh_to_sdf(convex_hull_mesh, np.array(self.get_centers()))
        self.update_occupancy_using_sdf(signs)

    def sample_point_cloud(self, pc_size):
        occupied_tets = self.curr_tetrahedrons
        weights = [tet.volume().abs() * tet.occupancy for tet in occupied_tets]
        weights_total = sum(weights)

        points_count = [np.int(np.ceil(((weight / weights_total) * pc_size).item())) for weight in weights]

        samples = []
        for i, tet in enumerate(occupied_tets):
            for _ in range(points_count[i]):
                r = np.random.rand(4)
                r /= np.sum(r)
                samples.append(sum([r[i] * tet.vertices[i].curr_loc for i in range(4)]))

        samples = random.choices(samples, k=pc_size)
        return torch.stack(samples)

    def sample_point_cloud_2(self, pc_size):
        centers = self.get_occupied_centers()
        vertices = []
        tmp = set()
        for tet in self.curr_tetrahedrons:
            if tet.occupancy > 0.5:
                for v in tet.vertices:
                    if v.get_original_xyz() not in tmp:
                        vertices.append(v)
                        tmp.add(v.get_original_xyz())
        vertices = torch.stack([v.curr_loc for v in self.vertices])
        samples = torch.cat([centers, vertices], dim=0)
        # samples = random.choices(samples, k=pc_size)
        # return torch.stack(samples)
        return samples

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

        self.init_setup(device)

    def do_init_calculations(self):
        print('Calculating neighborhoods')
        self.calculate_and_update_neighborhood()
        # print('Filling neighbors')
        # self.fill_neighbors()
        print('Merge same vertices')
        self.merge_same_vertices()

    def init_half_faces(self):
        for tet in self.curr_tetrahedrons:
            tet.calculate_half_faces(force=True)

        for tet in self.curr_tetrahedrons:
            for hf in tet.half_faces:
                hf.set_orientation()

    def init_setup(self, device):
        for tet in self.curr_tetrahedrons:
            tet.occupancy = tet.occupancy.cpu()
            tet.features = tet.features.to(device)
            tet.tet_num = tet.tet_num.to(device)
            tet.prev_features = tet.prev_features.to(device)
            for i in range(4):
                tet.vertices[i].curr_loc = tet.vertices[i].curr_loc.cpu()

        for tet in self.curr_tetrahedrons:
            tet.features.requires_grad_()

        self.init_half_faces()

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
                    face_container = tets_intersection(tet, nei)
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
            if torch.cdist(tet.center().curr_loc.unsqueeze(0), cube_center) <= 0.4:
                tet.occupancy = torch.tensor(1.)
                tet.init_occupancy = tet.occupancy.clone()
            else:
                tet.occupancy = torch.tensor(0.)
                tet.init_occupancy = tet.occupancy.clone()

    def fill_torus(self, major_radius, minor_radius, center):
        def check_point(x, y, z):
            temp = torch.sqrt(x * x + y * y)
            temp -= major_radius
            temp *= temp
            temp += z * z
            return temp < (minor_radius * minor_radius)

        for tet in self.curr_tetrahedrons:
            point = tet.center().curr_loc - center
            if check_point(*point):
                tet.occupancy = torch.tensor(1.)
                tet.init_occupancy = tet.occupancy.clone()
            else:
                tet.occupancy = torch.tensor(0.)
                tet.init_occupancy = tet.occupancy.clone()

    def subdivide_tets(self, net):
        new_tetrahedrons = []
        vertices = []
        device = self.curr_tetrahedrons[0].features.device
        for i, tet in enumerate(self.curr_tetrahedrons):
            new_tets, center = tet.subdivide()
            new_vertices = [Vertex(*v.get_original_xyz()) for v in tet.vertices] + [center]
            vertices.extend(new_vertices)
            new_tetrahedrons.extend(new_tets)
        self.vertices = vertices
        self.curr_tetrahedrons = new_tetrahedrons

        # neighbors + half faces calculations
        self.do_init_calculations()
        self.init_setup(device)
        self.init_half_faces()

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
            tet.set_as_init_values()

    def update_occupancy_using_sdf(self, sdf):
        for i, tet in enumerate(self.curr_tetrahedrons):
            if sdf[i] <= 0:
                tet.occupancy = torch.tensor(1.)
            else:
                tet.occupancy = torch.tensor(0.)
        for tet in self.curr_tetrahedrons:
            tet.set_as_init_values()

    def update_occupancy_from_filled_point_cloud(self, filled_pc):

        tets_n_points_inside = []

        for tet in self.curr_tetrahedrons:
            hfs_io_points = [(h.plane.coefficients[:-1] * filled_pc).sum(-1) + h.plane.coefficients[-1] <= 0 for h in
                             tet.half_faces]
            io_points = hfs_io_points[0]
            for i in range(1, 4):
                io_points = io_points * hfs_io_points[i]
            tets_n_points_inside.append(io_points.sum() / len(io_points))

        tets_n_points_inside = torch.stack(tets_n_points_inside)
        a = sorted(set([x.item() for x in tets_n_points_inside]))
        threshold = a[-1] * 0.2  # for noise canceling - take every tet that is 20% filled
        occupancies = (tets_n_points_inside >= threshold)
        for i, tet in enumerate(self.curr_tetrahedrons):
            tet.occupancy = occupancies[i]
            tet.set_as_init_values()

    def __getitem__(self, index):
        return self.curr_tetrahedrons[index]


if __name__ == '__main__':
    a = QuarTet('../objects/cube_0.1.tet', device='cuda')
    b = PointCloud()
    b.load_file('../objects/filled_g.obj')
    a.update_occupancy_from_filled_point_cloud(b.points)
    a.export_mesh('./try.obj')
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

if __name__ == '__main_1_':
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
