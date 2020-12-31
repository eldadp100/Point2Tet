import numpy as np
import torch
import random


def tensors_eq(v1, v2):
    return len(v1) == len(v2) and (v1 == v2).sum() == len(v2)


class Tetrahedron:
    def __init__(self, vertices, depth=0):
        self.vertices = vertices
        self.occupancy = torch.rand(1)  # very small chance to all be 0
        self.neighborhood = set()
        self.features = torch.stack([v.loc for v in self.vertices]).permute(1, 0).sum(dim=-1) / 4.
        self.prev_features = self.features
        self.sub_divided = None
        self.pooled = False
        self.depth = depth

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

    def get_center(self):
        a = torch.stack([v.loc for v in self.vertices])
        loc = a.permute(1, 0).sum(dim=1) / 4.
        return Vertex(loc[0], loc[1], loc[2])

    def __hash__(self):
        return (self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]).__hash__()

    def update_by_deltas(self, vertices_deltas):
        for i, v in enumerate(self.vertices):
            v.update_vertex(vertices_deltas[i])

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
            vert.update_vertex(vec)

    def get_diff_occupancy(self):
        return self.occupancy + 0.05


def calculate_and_update_neighborhood(list_of_tetrahedrons):
    for tet1 in list_of_tetrahedrons:
        for tet2 in list_of_tetrahedrons:
            if tet1.is_neighbor(tet2):
                tet1.add_neighbor(tet2)
                tet2.add_neighbor(tet1)


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

    def update_vertex(self, move_vector):
        self.loc = self.loc + move_vector

    def __hash__(self):
        x, y, z = self.loc[0].item(), self.loc[1].item(), self.loc[2].item()
        return (x, y, z).__hash__()


class UnitCube:
    def __init__(self, pos):
        self.pos = pos

    def divide_to_24(self):
        # (000) (010) (001) (011) --> (0 0.5 0.5)
        tri1 = Tetrahedron([Vertex(0, 0, 0), Vertex(0, 1, 0), Vertex(0, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)])
        tri2 = Tetrahedron([Vertex(0, 0, 0), Vertex(0, 0, 1), Vertex(0, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)])
        tri3 = Tetrahedron([Vertex(0, 1, 1), Vertex(0, 1, 0), Vertex(0, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)])
        tri4 = Tetrahedron([Vertex(0, 1, 1), Vertex(0, 0, 1), Vertex(0, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)])

        # (000) (100) (010) (110) --> (0.5 0.5 0)
        tri5 = Tetrahedron([Vertex(0, 0, 0), Vertex(1, 0, 0), Vertex(0.5, 0.5, 0), Vertex(0.5, 0.5, 0.5)])
        tri6 = Tetrahedron([Vertex(0, 0, 0), Vertex(0, 1, 0), Vertex(0.5, 0.5, 0), Vertex(0.5, 0.5, 0.5)])
        tri7 = Tetrahedron([Vertex(1, 1, 0), Vertex(1, 0, 0), Vertex(0.5, 0.5, 0), Vertex(0.5, 0.5, 0.5)])
        tri8 = Tetrahedron([Vertex(1, 1, 0), Vertex(0, 1, 0), Vertex(0.5, 0.5, 0), Vertex(0.5, 0.5, 0.5)])

        # (000) (100) (001) (101) --> (0.5 0 0.5)
        tri9 = Tetrahedron([Vertex(0, 0, 0), Vertex(1, 0, 0), Vertex(0.5, 0, 0.5), Vertex(0.5, 0.5, 0.5)])
        tri10 = Tetrahedron([Vertex(0, 0, 0), Vertex(0, 0, 1), Vertex(0.5, 0, 0.5), Vertex(0.5, 0.5, 0.5)])
        tri11 = Tetrahedron([Vertex(1, 0, 1), Vertex(1, 0, 0), Vertex(0.5, 0, 0.5), Vertex(0.5, 0.5, 0.5)])
        tri12 = Tetrahedron([Vertex(1, 0, 1), Vertex(0, 0, 1), Vertex(0.5, 0, 0.5), Vertex(0.5, 0.5, 0.5)])

        # (111) (011) (110) (010) --> (1 0.5 0.5)
        tri13 = Tetrahedron([Vertex(1, 1, 1), Vertex(1, 1, 0), Vertex(0.5, 1, 0.5), Vertex(0.5, 0.5, 0.5)])
        tri14 = Tetrahedron([Vertex(1, 1, 1), Vertex(0, 1, 1), Vertex(0.5, 1, 0.5), Vertex(0.5, 0.5, 0.5)])
        tri15 = Tetrahedron([Vertex(0, 1, 0), Vertex(1, 1, 0), Vertex(0.5, 1, 0.5), Vertex(0.5, 0.5, 0.5)])
        tri16 = Tetrahedron([Vertex(0, 1, 0), Vertex(0, 1, 1), Vertex(0.5, 1, 0.5), Vertex(0.5, 0.5, 0.5)])

        # (111) (101) (110) (100) --> (0.5 0.5 1)
        tri17 = Tetrahedron([Vertex(1, 1, 1), Vertex(1, 0, 1), Vertex(1, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)])
        tri18 = Tetrahedron([Vertex(1, 1, 1), Vertex(1, 1, 0), Vertex(1, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)])
        tri19 = Tetrahedron([Vertex(1, 0, 0), Vertex(1, 0, 1), Vertex(1, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)])
        tri20 = Tetrahedron([Vertex(1, 0, 0), Vertex(1, 1, 0), Vertex(1, 0.5, 0.5), Vertex(0.5, 0.5, 0.5)])

        # (111) (101) (011) (001) --> (0.5 1 0.5)
        tri21 = Tetrahedron([Vertex(1, 1, 1), Vertex(1, 0, 1), Vertex(0.5, 0.5, 1), Vertex(0.5, 0.5, 0.5)])
        tri22 = Tetrahedron([Vertex(1, 1, 1), Vertex(0, 1, 1), Vertex(0.5, 0.5, 1), Vertex(0.5, 0.5, 0.5)])
        tri23 = Tetrahedron([Vertex(0, 0, 1), Vertex(1, 0, 1), Vertex(0.5, 0.5, 1), Vertex(0.5, 0.5, 0.5)])
        tri24 = Tetrahedron([Vertex(0, 0, 1), Vertex(0, 1, 1), Vertex(0.5, 0.5, 1), Vertex(0.5, 0.5, 0.5)])

        tets = [tri1, tri2, tri3, tri4, tri5, tri6, tri7, tri8, tri9, tri10, tri11, tri12, tri13, tri14, tri15,
                tri16, tri17, tri18, tri19, tri20, tri21, tri22, tri23, tri24]

        for tet in tets:
            tet.translate(self.pos)

        return tets


class QuarTet:
    def __init__(self, n, device="cpu"):
        self.curr_tetrahedrons = []
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    pos = torch.tensor([x, y, z])
                    tets = UnitCube(pos).divide_to_24()
                    self.curr_tetrahedrons.extend(tets)

        calculate_and_update_neighborhood(self.curr_tetrahedrons)
        self.fill_neighbors()
        self.merge_same_vertices()

        for tet in self.curr_tetrahedrons:
            tet.occupancy = tet.occupancy.to(device)
            tet.features = tet.features.to(device)
            tet.prev_features = tet.prev_features.to(device)
            for i in range(4):
                tet.vertices[i].loc = tet.vertices[i].loc.to(device)

        for tet in self.curr_tetrahedrons:
            tet.features.requires_grad_()

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

    def init_occupancy_with_SDF(self, SDF):
        # TODO: that will improve results
        pass

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

    def export(self, path):
        pass

    def create_mesh(self):
        faces = list()
        for tet in self.curr_tetrahedrons:
            for nei in tet.neighborhood:
                if tet == nei:
                    continue
                if (tet.occupancy > 0.5) ^ (nei.occupancy > 0.5):
                    face = Face(tet, nei)
                    faces.append(face)

    def export_mesh(self, path):
        pass


if __name__ == '__main__':
    a = QuarTet(2, 'cpu')
    for tet in a:
        print(len(tet.neighborhood))
    b = a.sample_disjoint_faces(4)
    c = a.get_occupied_tets()
    d = a.sample_point_cloud(100)
    print(b)
    print(c)
    print(d)
