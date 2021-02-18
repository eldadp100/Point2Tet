import time

import numpy as np
import torch
import random
from pointcloud import PointCloud
import itertools

from tetrahedral_group import TetsGroupSharesVertex


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
        self.init_vertices = [v.clone() for v in self.vertices]
        self.init_occupancy = self.occupancy.clone()

    def sample_points(self, n):
        a, b, c, d = [v.loc for v in self.vertices]
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

    def update_occupancy(self, new_occupancy):
        self.occupancy = new_occupancy

    def is_neighbor(self, other):
        c = 0
        for v1 in self.vertices:
            for v2 in other.vertices:
                if tensors_eq(v1.loc, v2.loc):
                    c += 1
        return c == 3

    def center(self):
        a = torch.stack([v.loc for v in self.vertices])
        loc = a.permute(1, 0).sum(dim=1) / 4.
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
        p1, p2, p3, p4 = [v.loc for v in self.vertices]
        return torch.det(torch.stack([p1 - p4, p2 - p4, p3 - p4])) / 6

    def translate(self, vec):
        for vert in self.vertices:
            vert.update_vertex(vec)

    def reset(self):
        self.features = self.init_features.clone().to(self.features.device)
        self.vertices = [v.clone() for v in self.init_vertices]
        self.occupancy = self.init_occupancy.clone()

    def get_faces(self):
        return list(itertools.combinations(self.vertices, 3))

    def calculate_half_faces(self, force=False):
        if force:
            self.half_faces = []
        assert len(self.half_faces) == 0
        for nei in self.neighborhood:
            if self.is_neighbor(nei):  # can also be on the boundary so we add itself as a neighbor
                face_coords = intersect(self, nei)
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
            self.half_faces.append(HalfFace([v.loc for v in coords], (self, self)))

        ##########################################################################
        for v in self.vertices:
            self.faces_by_vertex[v] = [hf for hf in self.half_faces if hf.has(v)]
            tmp = [hf for hf in self.half_faces if not hf.has(v)]
            assert len(tmp) == 1
            self.faces_by_vertex_opposite[v] = tmp[0]

    def get_half_faces(self):
        return self.half_faces


def calculate_and_update_neighborhood(tetrahedrons, vertices):
    vertices_to_tets_dict = dict()
    for vertex in vertices:
        vertices_to_tets_dict[vertex] = set()
    for tet in tetrahedrons:
        for vertex in tet:
            vertices_to_tets_dict[vertex].add(tet)

    for tet in tetrahedrons:
        for face in tet.get_faces():
            neighbor_set = set.intersection(*[vertices_to_tets_dict[vertex] for vertex in face])
            assert (len(neighbor_set) == 1 or len(neighbor_set) == 2)
            for neighbor in neighbor_set:
                if neighbor != tet:
                    neighbor.add_neighbor(tet)
                    tet.add_neighbor(neighbor)


def vertices_intersection(ver1, ver2):
    intersection = []
    for v1 in ver1:
        exist = False
        for v2 in ver2:
            if tensors_eq(v1.loc, v2.loc):
                exist = True
        if exist:
            intersection.append(v1.loc)
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

    def get_normal(self):
        return torch.tensor([self.a, self.b, self.c])

    def get_point_side(self, x):
        """ True is right, False is left"""
        return self.a * x[0] + self.b * x[1] + self.c * x[2] + self.d > 0.

    def signed_distance(self, x, side):
        val = self.a * x[0] + self.b * x[1] + self.c * x[2] + self.d
        if side:
            return val
        else:
            return -val


class HalfFace:
    def __init__(self, coords, tets):
        self.coords = coords

        # calculate plane according to 3 points
        mat = np.array([c.numpy() for c in coords])
        if np.linalg.matrix_rank(mat) == 3:
            b = -np.ones(3)
            solution = np.linalg.solve(mat, b)
            self.plane = Plane(solution[0], solution[1], solution[2], 0.)
        else:
            solution = np.linalg.eig(mat)
            solution = solution[1][abs(solution[0]).argmin()]
            self.plane = Plane(solution[0], solution[1], solution[2], 1.)

        self.tets = tets
        self.oriented = False  # all half faces of the same tet are with same orientation

    def set_orientation(self):
        if self.oriented:
            return None
        tet_half_faces = self.tets[0].get_half_faces()
        n1 = tet_half_faces[0].plane.get_normal()
        for half_face in tet_half_faces:
            if not torch.dot(half_face.plane.get_normal(), n1) > 0:
                half_face.plane.change_orientation()
        self.oriented = True

    def is_eq(self, three_vertices):
        return len(vertices_intersection(three_vertices, [Vertex(*x) for x in self.coords])) == 3

    def get_neighbor(self):
        return self.tets[1]

    def has(self, v):
        for v2 in self.coords:
            if tensors_eq(v.loc, v2):
                return True

        return False


class Vertex:
    def __init__(self, x, y, z):

        self.loc = torch.tensor([x, y, z], dtype=torch.float32)
        self.original_loc = self.loc.detach().clone()
        self.on_boundary = x == 0 or x == 1 or y == 0 or y == 1 or z == 0 or z == 1

        self.tets_group = None
        self.last_update_signed_distance = None

    def update_sd_loss(self, move_vector):
        if not self.on_boundary:
            self.last_update_signed_distance += self.tets_group.query_direction(move_vector)

    def update_vertex(self, move_vector):
        if not self.on_boundary:
            # self.loc = torch.clip(self.loc + move_vector, 0., 1.)
            self.loc = self.loc + move_vector

        self.tets_group.v = self

    def get_xyz(self):
        x, y, z = self.loc[0].item(), self.loc[1].item(), self.loc[2].item()
        return x, y, z

    def clone(self):
        v = Vertex(*self.get_xyz())
        v.loc = v.loc.to(self.loc.device)
        v.tets_group = self.tets_group
        return v

    def __hash__(self):
        x, y, z = self.loc[0].item(), self.loc[1].item(), self.loc[2].item()
        return (x, y, z).__hash__()

    def __ge__(self, other):
        x, y, z = self.loc
        ox, oy, oz = other.loc
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
    def __init__(self, path='../cube_0.05.tet', device='cpu'):
        self.curr_tetrahedrons = None
        self.vertices = None
        self.load(path, device)

        self.last_vertex_update_average = None

    def fill_neighbors(self):
        for tet in self.curr_tetrahedrons:
            for i in range(4 - len(tet.neighborhood)):
                tet.add_neighbor(tet)

    def merge_same_vertices(self):
        all_vertices = {}
        for tet in self.curr_tetrahedrons:
            for v in tet.vertices:
                all_vertices[v] = v

        for tet in self.curr_tetrahedrons:
            new_vertices = []
            for v in tet.vertices:
                new_vertices.append(all_vertices[v])
            tet.vertices = new_vertices

    def zero_grad(self):
        for tet in self.curr_tetrahedrons:
            tet.features = tet.features.detach().clone()
            tet.prev_features = tet.prev_features.detach().clone()
            for i in range(4):
                tet.vertices[i].loc = tet.vertices[i].loc.detach().clone()

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
        samples_weights = []
        for tet in self.curr_tetrahedrons:
            samples_weights.append((tet.center().loc, tet.occupancy))  # grad of 1
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
                samples.append(sum([r[i] * tet.vertices[i].loc for i in range(4)]))

        # if len(samples) == 0:
        #     return torch.rand(pc_size, 3)
        samples = random.choices(samples, k=pc_size)
        # return torch.stack(samples), volumes
        return torch.stack(samples)

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

    def export(self, path):
        with open(path, "w") as output_file:
            vertex_to_idx = {}
            n_ver = 0
            to_write = ""
            for tet in self.curr_tetrahedrons:
                for v in tet.vertices:
                    if v not in vertex_to_idx:
                        x, y, z = v.loc
                        to_write += f"{x} {y} {z}\n"
                        vertex_to_idx[v] = n_ver
                        n_ver += 1

            for tet in self.curr_tetrahedrons:
                if tet.occupancy > 0.5:
                    indices = [vertex_to_idx[v] for v in tet.vertices]
                    to_write += f"{indices[0]} {indices[1]} {indices[2]} {indices[3]}\n"

            output_file.write(f"tet {n_ver} {len(self.curr_tetrahedrons)}\n")
            output_file.write(to_write)

    def load(self, path, device):
        self.curr_tetrahedrons = []
        vertices = []
        with open(path, "r") as input_file:
            first_line = input_file.readline()
            first_line_split = first_line.split(' ')
            print(f'Loading tet file\nReading {int(first_line_split[1])} vertices')
            for i in range(int(first_line_split[1])):
                line = input_file.readline()
                coordinates = line.split(' ')
                vertices.append(Vertex(*[float(c) for c in coordinates]))
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

        print('Calculating neighborhoods')
        calculate_and_update_neighborhood(self.curr_tetrahedrons, vertices)
        print('Filling neighbors')
        self.fill_neighbors()
        print('Merge same vertices')
        self.merge_same_vertices()

        for tet in self.curr_tetrahedrons:
            tet.occupancy = tet.occupancy.cpu()
            tet.features = tet.features.to(device)
            tet.tet_num = tet.tet_num.to(device)
            tet.prev_features = tet.prev_features.to(device)
            for i in range(4):
                tet.vertices[i].loc = tet.vertices[i].loc.cpu()

        for tet in self.curr_tetrahedrons:
            tet.features.requires_grad_()

        for tet in self.curr_tetrahedrons:
            tet.calculate_half_faces()

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
                    face_coords = intersect(tet, nei)
                    if face_coords not in faces:
                        faces.add(face_coords)
            # print(tet.occupancy)
        return faces

    def export_mesh(self, path):
        faces = self.create_mesh()
        obj_file_str_vert = []
        obj_file_str_faces = []
        vertices = {}
        c = 1
        for i, f_coords in enumerate(faces):
            for v in f_coords:
                x, y, z = v.get_xyz()
                if (x, y, z) not in vertices:
                    vertices[(x, y, z)] = c
                    c += 1
                obj_file_str_vert.append(f"v {x} {y} {z}")
            obj_file_str_faces.append(
                f"f {vertices[f_coords[0].get_xyz()]} {vertices[f_coords[1].get_xyz()]} {vertices[f_coords[2].get_xyz()]}")
        with open(path, 'w+') as f:
            f.write("\n".join(obj_file_str_vert))
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
            if torch.cdist(tet.center().loc.unsqueeze(0), cube_center) <= 0.2:
                tet.occupancy = torch.tensor(1.)
                tet.init_occupancy = tet.occupancy.clone()
            else:
                tet.occupancy = torch.tensor(0.)
                tet.init_occupancy = tet.occupancy.clone()


if __name__ == '__main_1_':
    # a = QuarTet(2, 'cpu')
    a = QuarTet('../objects/cube_0.1.tet')
    a.fill_sphere()
    a.export_point_cloud('./pc.obj', 10000)

    pc = PointCloud()
    pc.load_file('./filled_sphere.obj')

if __name__ == '__main__':
    a = QuarTet('../checkpoints/default_name/quartet_100.tet')
    a.export_mesh('./mesh.obj')
    a.export_point_cloud('./pc.obj', 10000)
    print(a.vertices)
