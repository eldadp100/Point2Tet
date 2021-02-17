import time

import numpy as np
import torch
import random
from pointcloud import PointCloud
import itertools
from fill_pointcloud import face_ray_intersect
import os


def tensors_eq(v1, v2):
    return len(v1) == len(v2) and (v1 == v2).sum() == len(v2)


class Tetrahedron:
    def __init__(self, vertices, depth=0):
        self.vertices = sorted(vertices)
        self.occupancy = torch.tensor(1./6000)  # torch.rand(1)  # very small chance to all be 0
        self.neighborhood = set()
        self.features = torch.stack([v.loc for v in self.vertices]).permute(1, 0).sum(dim=-1) / 4.
        self.features = torch.rand(30)
        # rand_vec = torch.rand(3) - 1
        # self.features = torch.stack([v.loc for v in self.vertices]).permute(1, 0).sum(dim=-1) / 4. + rand_vec
        # self.features = torch.cat([self.features, torch.rand(27, requires_grad=True)])
        self.prev_features = self.features
        self.sub_divided = None
        self.pooled = False
        self.depth = depth

        self.init_features = self.features.clone()
        self.init_vertices = [v for v in self.vertices]
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
        '''
        :return: returns a list of 4 tuples of size 3 of vertices, each representing a face
        '''
        return list(map(set, itertools.combinations(self.vertices, 3)))


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


def intersect(tet1, tet2):
    intersection = []
    for v1 in tet1.vertices:
        exist = False
        for v2 in tet2.vertices:
            if tensors_eq(v1.loc, v2.loc):
                exist = True
        if exist:
            intersection.append(v1)
    return intersection


class Face:
    def __init__(self, tet1=None, tet2=None, vertices=None):
        if vertices is not None:
            self.face_coords = torch.stack([v.loc for v in vertices])
        else:
            assert tet1.is_neighbor(tet2)
            self.tet1 = tet1
            self.tet2 = tet2
            self.face_coords = intersect(self.tet1, self.tet2)
        self.plane = None

    def get_tets(self):
        return self.tet1, self.tet2

    def __iter__(self):
        return iter(self.face_coords)

    def get_face_plane(self):
        if self.plane is not None:
            return self.plane
        # lazy initialization
        mat = np.ones((4, 4))
        mat[:3, :3] = np.array(self.face_coords)
        mat[3, 3] = 0
        b = np.zeros(4)
        b[3] = -1
        solution = np.linalg.solve(mat, b)
        self.plane = solution
        return solution


class Vertex:
    def __init__(self, x, y, z, neighborhood=[]):
        self.loc = torch.tensor([x, y, z], dtype=torch.float64)
        self.on_boundary = x == 0 or x == 1 or y == 0 or y == 1 or z == 0 or z == 1
        self.neighborhood = neighborhood

    def update_vertex(self, move_vector):
        if not self.on_boundary:
            ####### FACE CLIPPING OF VERTICES - NOT EFFICIENT! #######
            # if self.neighborhood is not None:
            #     for face in self.neighborhood:
            #         intersects, intersections = face_ray_intersect((self.loc.unsqueeze(0), move_vector.type(torch.float64)),
            #                                                        torch.tensor(face.face_coords, dtype=torch.float64), face.get_face_plane())
            #         if intersects[0] and torch.cdist(intersections[0].unsqueeze(0),
            #                                          self.loc.unsqueeze(0)) >= torch.linalg.norm(move_vector):
            #             self.loc = torch.clip(self.loc + move_vector, 0., 1.)
            # else:
            self.loc = torch.clip(self.loc + move_vector, 0., 1.)

    def get_xyz(self):
        x, y, z = self.loc[0].item(), self.loc[1].item(), self.loc[2].item()
        return x, y, z

    def clone(self):
        v = Vertex(*self.get_xyz())
        v.loc = v.loc.to(self.loc.device)
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


class QuarTet:
    def __init__(self, path='../cube_0.05.tet', device='cpu', occupancies_path=None):
        self.curr_tetrahedrons = None
        self.vertices = None
        self.load(path, device, occupancies_path)
        self.fill_vertex_neighbors()

    def fill_vertex_neighbors(self):
        for tet in self.curr_tetrahedrons:
            faces = tet.get_faces()
            for v in tet.vertices:
                for face in faces:
                    if v not in face:
                        v.neighborhood.append(Face(vertices=face))

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

    def sample_disjoint_faces(self, N):  # TODO: do it exact N
        faces = []
        visited_tets = set()
        sampled_indices = np.random.randint(0, len(self.curr_tetrahedrons), size=N)
        for idx in sampled_indices:
            tet = self.curr_tetrahedrons[idx]
            if tet in visited_tets:
                continue

            neighbor_idx = np.random.randint(0, len(tet.neighborhood))
            neighbor = list(tet.neighborhood)[neighbor_idx]
            if neighbor != tet:
                visited_tets.add(tet)
                visited_tets.add(neighbor)
                faces.append(Face(tet, neighbor))

        return faces

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

    def sample_point_cloud(self, pc_size):

        samples_weights = []
        for tet in self.curr_tetrahedrons:
            samples_weights.append((tet.center().loc, tet.occupancy))  # grad of 1
        samples_weights = random.choices(samples_weights, k=pc_size)
        samples = torch.stack([x[0] for x in samples_weights])
        weights = torch.stack([x[1] for x in samples_weights])
        return samples, weights

    def sample_point_cloud_2(self, pc_size):
        occupied_tets = self.curr_tetrahedrons
        volumes = [tet.volume() * tet.occupancy for tet in occupied_tets]
        volumes_total = sum(volumes)

        points_count = [np.int(np.ceil(((volume / volumes_total) * pc_size).item())) for volume in volumes]

        samples = []
        for i, tet in enumerate(occupied_tets):
            samples.extend(tet.sample_points(points_count[i]))

        samples = random.choices(samples, k=pc_size)
        return torch.stack(samples)

    def sample_point_cloud_3(self, pc_size):
        occupied_tets = self.curr_tetrahedrons
        volumes = [tet.volume() * tet.occupancy for tet in occupied_tets]
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

    def sample_point_cloud_4(self, pc_size):
        occupied_tets = self.curr_tetrahedrons
        # volumes = [tet.volume() * tet.occupancy for tet in occupied_tets]
        volumes = [(tet.occupancy > 0.5) + 0.01 for tet in occupied_tets]
        volumes_total = sum(volumes)

        points_count = [np.int(np.ceil(((volume / volumes_total) * pc_size).item())) for volume in volumes]

        samples = []
        for i, tet in enumerate(occupied_tets):
            little_qube = (torch.rand(points_count[i], 3) - 0.5) / 50
            little_qube = little_qube + tet.center().loc.expand_as(little_qube)
            samples.append(little_qube)
        return torch.cat(samples)

    def export(self, path, export_occupancies=True):
        """
        TODO: change to .tet format
        """
        with open(path, "w") as output_file:
            vertex_to_idx = {}
            n_ver = 0
            to_write = []
            occupancies_str = []
            for tet in self.curr_tetrahedrons:
                for v in tet.vertices:
                    if v not in vertex_to_idx:
                        x, y, z = v.loc
                        to_write.append(f"{x} {y} {z}\n")
                        vertex_to_idx[v] = n_ver
                        n_ver += 1

            for tet in self.curr_tetrahedrons:
                indices = [vertex_to_idx[v] for v in tet.vertices]
                to_write.append(f"{indices[0]} {indices[1]} {indices[2]} {indices[3]}\n")
                occupancies_str.append(f"{tet.occupancy}\n")

            output_file.write(f"tet {n_ver} {len(self.curr_tetrahedrons)}\n")
            output_file.write(''.join(to_write))

        name_without_extension, _ = os.path.splitext(path)
        with open(f'{name_without_extension}_occupancies.occ', 'w') as output_file:
            output_file.write(''.join(occupancies_str))

    def load(self, path, device, occupancies_path=None):
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
            num_tets = int(first_line_split[2])
            print(f'Finished reading vertices\nReading {num_tets} tetrahedrons')
            for i in range(num_tets):
                line = input_file.readline()
                vertex_indices = line.split(' ')
                if len(vertex_indices) < 3:
                    continue
                self.curr_tetrahedrons.append(Tetrahedron([vertices[int(i)] for i in vertex_indices]))
                if i % 2000 == 0 and i > 0:
                    print(f'Read {i} tetrhedrons')

        print('Calculating neighborhoods')
        calculate_and_update_neighborhood(self.curr_tetrahedrons, vertices)
        print('Filling neighbors')
        self.fill_neighbors()
        # print('Merge same vertices')
        # self.merge_same_vertices()

        if occupancies_path is None:
            base_name = os.path.splitext(path)[0]
            occupancies_path = f"{base_name}_occupancies.occ"

        if os.path.exists(occupancies_path):
            with open(occupancies_path, 'r') as occupancies_input:
                for occ, tet in zip(occupancies_input, self.curr_tetrahedrons):
                    tet.occupancy = torch.tensor(float(occ))

        for tet in self.curr_tetrahedrons:
            tet.occupancy = tet.occupancy.cpu()
            tet.features = tet.features.to(device)
            tet.prev_features = tet.prev_features.to(device)
            for i in range(4):
                tet.vertices[i].loc = tet.vertices[i].loc.cpu()

        for tet in self.curr_tetrahedrons:
            tet.features.requires_grad_()

    def reset(self):
        for tet in self.curr_tetrahedrons:
            tet.reset()

    def create_mesh(self):
        faces = list()
        for tet in self.curr_tetrahedrons:
            for nei in tet.neighborhood:
                if tet == nei:
                    continue
                if (tet.occupancy > 0.5) ^ (nei.occupancy > 0.5):
                    face = Face(tet, nei)
                    faces.append(face)
            # print(tet.occupancy)
        return faces

    def export_mesh(self, path):
        faces = self.create_mesh()
        obj_file_str_vert = []
        obj_file_str_faces = []
        vertices = {}
        c = 1
        for i, f in enumerate(faces):
            f_coords = f.face_coords
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
        points = self.sample_point_cloud_4(n)
        print(f"Done {time.time() - s}")
        pc = PointCloud()
        pc.init_with_points(points)
        pc.write_to_file(path)

    def fill_sphere(self):
        cube_center = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
        for tet in self.curr_tetrahedrons:
            if torch.cdist(tet.center().loc.unsqueeze(0), cube_center) <= 0.5:
                tet.occupancy = torch.tensor(1.)
            else:
                tet.occupancy = torch.tensor(0.)


if __name__ == '__main__':
    # a = QuarTet(2, 'cpu')
    # a = QuarTet('../objects_old/cube_0.05.tet')
    a = QuarTet('try.tet')
    # a.fill_sphere()
    for tet in a:
        print(len(tet.neighborhood))
    # a.fill_sphere()
    # a.export('try.tet')
    # a = QuarTet('./quartet_100.tet')
    # for tet in a:
    #     print(len(tet.neighborhood))
    # a.fill_sphere()
    # a.export_point_cloud('./pc.obj', 10000)

    # pc = PointCloud()
    # pc.load_file('./filled_sphere.obj')
