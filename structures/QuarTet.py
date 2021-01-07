import numpy as np
import torch
import random

class QuarTet:
    NEIGHBORS_COUNT = 4

    def __init__(self, divide_count=1, device="cpu"):
        self.device = device
        self.curr_tetrahedrons = self.sub_divide(divide_count)

        self.calculate_and_update_neighborhood()
        self.fill_neighbors()
        self.merge_same_vertices()

        for tet in self.curr_tetrahedrons:
            tet.features.requires_grad_()

    def __iter__(self):
        return self.curr_tetrahedrons.__iter__()

    def sub_divide(self, n):
        tets = []
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    pos = torch.tensor([x, y, z])
                    current_tets = cube_to_24_tets(pos, self.device)
                    tets.extend(current_tets)
        return tets

    def calculate_and_update_neighborhood(self):
        for tet1 in self.curr_tetrahedrons:
            for tet2 in self.curr_tetrahedrons:
                if tet1.is_neighbor(tet2):
                    tet1.add_neighbor(tet2)
                    tet2.add_neighbor(tet1)

    def fill_neighbors(self):
        for tet in self.curr_tetrahedrons:
            while len(tet.neighborhood) < self.NEIGHBORS_COUNT:
                tet.add_neighbor(tet)

    def merge_same_vertices(self):
        all_vertices = {}

        # Save one copy of each vertex
        for tet in self.curr_tetrahedrons:
            for v in tet.vertices:
                all_vertices[v] = v

        # Update the tets to use the shared vertices
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

    def get_occupied_tets(self):
        result = []
        for tet in self:
            if tet.occupancy > 0:
                result.append(tet)
        return result

    def sample_point_cloud(self, pc_size):
        samples = []
        occupied_tets = self.get_occupied_tets()
        volumes = [tet.volume() * tet.get_diff_occupancy() for tet in occupied_tets]
        volumes_total = sum(volumes)

        points_count = [np.int(np.ceil(((volume / volumes_total) * pc_size).item())) for volume in volumes]

        for i, tet in enumerate(occupied_tets):
            for _ in range(points_count[i]):
                samples.append(sum([vertex.loc * np.random.uniform(0, 1) for vertex in tet.vertices]))

        samples = random.choices(samples, k=pc_size)
        return torch.stack(samples)

    def reset(self):
        for tet in self.curr_tetrahedrons:
            tet.reset()

    def create_surface_mesh(self):
        faces = list()
        for tet in self.curr_tetrahedrons:
            for nei in tet.neighborhood:
                if tet == nei:
                    continue
                if (tet.occupancy > 0.5) ^ (nei.occupancy > 0.5):
                    face = Face(tet, nei)
                    faces.append(face)
        return faces

    def export_surface_mesh(self, path):
        faces = self.create_surface_mesh()
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
                obj_file_str_vert.append(f"v {x} {y} {z}\n")
            obj_file_str_faces.append(
                f"f {vertices[f_coords[0].get_xyz()]} {vertices[f_coords[1].get_xyz()]} {vertices[f_coords[2].get_xyz()]}\n")
        with open(path, 'w+') as f:
            f.write("\n".join(obj_file_str_vert))
            f.write("\n".join(obj_file_str_faces))

    def export(self, path):
        with open(path, "w") as output_file:
            vertex_to_idx = {}
            i = 0
            for tet in self.curr_tetrahedrons:
                for v in tet.vertices:
                    if v not in vertex_to_idx:
                        x, y, z = v.loc
                        output_file.write(f"v {x} {y} {z}\n")
                        vertex_to_idx[v] = i
                        i += 1

            for tet in self.curr_tetrahedrons:
                indices = [vertex_to_idx[v] for v in tet.vertices]
                output_file.write(f"f {indices[0]} {indices[1]} {indices[2]} {indices[3]}\n")


class Tetrahedron:
    TET_VERTEX_COUNT = 4

    def __init__(self, vertices, device):
        self.device = device
        self.vertices = self.init_vertices(vertices)
        self.features = self.init_features()
        self.prev_features = self.features
        self.occupancy = torch.rand(1).to(device)
        self.neighborhood = set()

    def __hash__(self):
        return (self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]).__hash__()

    def init_vertices(self, vertices=None):
        if vertices is not None:
            self.start_vertices = vertices

        return [v.clone().to(self.device) for v in self.start_vertices]

    def init_features(self):
        if not hasattr(self, "start_features"):
            self.start_features = torch.stack([v.loc for v in self.vertices]).permute(1, 0).sum(dim=-1) / 4.

        return self.start_features.clone().to(self.device)

    def add_neighbor(self, neighbor):
        self.neighborhood.add(neighbor)

    def update_occupancy(self, new_occupancy):
        self.occupancy = new_occupancy.to(self.device)

    def is_neighbor(self, other):
        matching_vertices = 0
        for v1 in self.vertices:
            for v2 in other.vertices:
                if tensors_eq(v1.loc, v2.loc):
                    matching_vertices += 1
                    if matching_vertices >= 3:
                        return True
        return False

    def get_center(self):
        a = torch.stack([v.loc for v in self.vertices])
        loc = a.permute(1, 0).sum(dim=1) / 4.
        return Vertex(loc[0], loc[1], loc[2])

    def update_by_deltas(self, vertices_deltas):
        for i, v in enumerate(self.vertices):
            v.translate(vertices_deltas[i])

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
        return (abs(self.determinant_3x3((
            self.subtract(p1, p2),
            self.subtract(p2, p3),
            self.subtract(p3, p4),
        ))) / 6.0)

    def translate(self, vec):
        for vert in self.vertices:
            vert.translate(vec)

    def get_diff_occupancy(self):
        return self.occupancy + 0.05

    def reset(self):
        self.features = self.init_features()
        self.vertices = self.init_vertices()


class Face:
    def __init__(self, tet1, tet2):
        assert tet1.is_neighbor(tet2)
        self.tet1 = tet1
        self.tet2 = tet2
        self.face_coords = intersect(self.tet1, self.tet2)

    def get_tets(self):
        return self.tet1, self.tet2


class Vertex:
    def __init__(self, x, y, z):
        self.loc = torch.tensor([x, y, z], dtype=torch.float32)

    def __hash__(self):
        x, y, z = self.loc[0].item(), self.loc[1].item(), self.loc[2].item()
        return (x, y, z).__hash__()

    def to(self, device):
        self.loc = self.loc.to(device)
        return self

    def translate(self, move_vector):
        self.loc = self.loc + move_vector

    def get_xyz(self):
        x, y, z = self.loc[0].item(), self.loc[1].item(), self.loc[2].item()
        return (x, y, z)

    def clone(self):
        return Vertex(*self.get_xyz())


def tensors_eq(v1, v2):
    return len(v1) == len(v2) and (v1 == v2).sum() == len(v2)

def cube_to_24_tets(pos, device):
    # (000) (010) (001) (011) --> (0 0.5 0.5)
    tri1 = Tetrahedron([Vertex(0, 0, 0), Vertex(0, 1, 0), Vertex(0, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)], device)
    tri2 = Tetrahedron([Vertex(0, 0, 0), Vertex(0, 0, 1), Vertex(0, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)], device)
    tri3 = Tetrahedron([Vertex(0, 1, 1), Vertex(0, 1, 0), Vertex(0, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)], device)
    tri4 = Tetrahedron([Vertex(0, 1, 1), Vertex(0, 0, 1), Vertex(0, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)], device)

    # (000) (100) (010) (110) --> (0.5 0.5 0)
    tri5 = Tetrahedron([Vertex(0, 0, 0), Vertex(1, 0, 0), Vertex(0.5, 0.5, 0), Vertex(0.5, 0.5, 0.5)], device)
    tri6 = Tetrahedron([Vertex(0, 0, 0), Vertex(0, 1, 0), Vertex(0.5, 0.5, 0), Vertex(0.5, 0.5, 0.5)], device)
    tri7 = Tetrahedron([Vertex(1, 1, 0), Vertex(1, 0, 0), Vertex(0.5, 0.5, 0), Vertex(0.5, 0.5, 0.5)], device)
    tri8 = Tetrahedron([Vertex(1, 1, 0), Vertex(0, 1, 0), Vertex(0.5, 0.5, 0), Vertex(0.5, 0.5, 0.5)], device)

    # (000) (100) (001) (101) --> (0.5 0 0.5)
    tri9 = Tetrahedron([Vertex(0, 0, 0), Vertex(1, 0, 0), Vertex(0.5, 0, 0.5), Vertex(0.5, 0.5, 0.5)], device)
    tri10 = Tetrahedron([Vertex(0, 0, 0), Vertex(0, 0, 1), Vertex(0.5, 0, 0.5), Vertex(0.5, 0.5, 0.5)], device)
    tri11 = Tetrahedron([Vertex(1, 0, 1), Vertex(1, 0, 0), Vertex(0.5, 0, 0.5), Vertex(0.5, 0.5, 0.5)], device)
    tri12 = Tetrahedron([Vertex(1, 0, 1), Vertex(0, 0, 1), Vertex(0.5, 0, 0.5), Vertex(0.5, 0.5, 0.5)], device)

    # (111) (011) (110) (010) --> (1 0.5 0.5)
    tri13 = Tetrahedron([Vertex(1, 1, 1), Vertex(1, 1, 0), Vertex(0.5, 1, 0.5), Vertex(0.5, 0.5, 0.5)], device)
    tri14 = Tetrahedron([Vertex(1, 1, 1), Vertex(0, 1, 1), Vertex(0.5, 1, 0.5), Vertex(0.5, 0.5, 0.5)], device)
    tri15 = Tetrahedron([Vertex(0, 1, 0), Vertex(1, 1, 0), Vertex(0.5, 1, 0.5), Vertex(0.5, 0.5, 0.5)], device)
    tri16 = Tetrahedron([Vertex(0, 1, 0), Vertex(0, 1, 1), Vertex(0.5, 1, 0.5), Vertex(0.5, 0.5, 0.5)], device)

    # (111) (101) (110) (100) --> (0.5 0.5 1)
    tri17 = Tetrahedron([Vertex(1, 1, 1), Vertex(1, 0, 1), Vertex(1, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)], device)
    tri18 = Tetrahedron([Vertex(1, 1, 1), Vertex(1, 1, 0), Vertex(1, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)], device)
    tri19 = Tetrahedron([Vertex(1, 0, 0), Vertex(1, 0, 1), Vertex(1, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)], device)
    tri20 = Tetrahedron([Vertex(1, 0, 0), Vertex(1, 1, 0), Vertex(1, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)], device)

    # (111) (101) (011) (001) --> (0.5 1 0.5)
    tri21 = Tetrahedron([Vertex(1, 1, 1), Vertex(1, 0, 1), Vertex(0.5, 0.5, 1), Vertex(0.5, 0.5, 0.5)], device)
    tri22 = Tetrahedron([Vertex(1, 1, 1), Vertex(0, 1, 1), Vertex(0.5, 0.5, 1), Vertex(0.5, 0.5, 0.5)], device)
    tri23 = Tetrahedron([Vertex(0, 0, 1), Vertex(1, 0, 1), Vertex(0.5, 0.5, 1), Vertex(0.5, 0.5, 0.5)], device)
    tri24 = Tetrahedron([Vertex(0, 0, 1), Vertex(0, 1, 1), Vertex(0.5, 0.5, 1), Vertex(0.5, 0.5, 0.5)], device)

    tets = [tri1, tri2, tri3, tri4, tri5, tri6, tri7, tri8, tri9, tri10, tri11, tri12, tri13, tri14, tri15,
            tri16, tri17, tri18, tri19, tri20, tri21, tri22, tri23, tri24]

    for tet in tets:
        tet.translate(pos)

    return tets

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