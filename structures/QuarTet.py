import torch
import numpy as np


class Tetrahedron:
    def __init__(self, vertices):
        self.vertices = vertices
        self.occupancy = np.random.choice([0, 1])  # very small chance to all be 0
        self.neighborhood = set()

        self.sub_divided = None

    def add_neighbor(self, neighbor):
        self.neighborhood.add(neighbor)

    def update_occupancy(self, new_occupancy):
        self.occupancy = new_occupancy

    def is_neighbor(self, other):
        c = 0
        for v1 in self.vertices:
            for v2 in other.vertices:
                if v1 == v2:
                    c += 1
        return c == 3

    def get_center(self):
        a = torch.stack([v.loc for v in self.vertices])
        return a.permute(1, 0).sum(dim=1) / 4

    def sub_divide(self):
        if self.sub_divided is not None:
            ret_tets = []
            center = self.get_center()
            for remove_vertex in self.vertices:
                new_tet = []
                for v in self.vertices:
                    if v != remove_vertex:
                        new_tet.append(v)
                new_tet.append(center)
                Tetrahedron(new_tet)
            self.sub_divided = ret_tets

        return self.sub_divided

    def update_after_all_finish_sub_divide(self):
        neighborhood_candidates = []
        for tet in self.neighborhood:
            neighborhood_candidates.extend(tet.sub_divided)

        calculate_and_update_neighborhood(neighborhood_candidates)


def calculate_and_update_neighborhood(list_of_tetrahedrons):
    for tet1 in list_of_tetrahedrons:
        for tet2 in list_of_tetrahedrons:
            if tet1.is_neighbor(tet2):
                tet1.add_neighbor(tet2)
                tet2.add_neighbor(tet1)


class Vertex:
    def __init__(self, x, y, z):
        self.loc = torch.tensor([x, y, z])
        self.features = self.loc  # TODO: maybe add more features (e.g. noraml)


class UnitCube:
    def __init__(self):
        pass

    def divide(self):
        tri1 = Tetrahedron([Vertex(0, 1, 0), Vertex(0, 1, 1), Vertex(1, 1, 1), Vertex(0, 0, 1)])
        tri2 = Tetrahedron([Vertex(0, 1, 0), Vertex(1, 1, 1), Vertex(1, 1, 0), Vertex(1, 0, 0)])
        tri3 = Tetrahedron([Vertex(0, 0, 0), Vertex(1, 0, 0), Vertex(0, 0, 1), Vertex(0, 1, 0)])
        tri4 = Tetrahedron([Vertex(1, 0, 0), Vertex(1, 0, 1), Vertex(0, 0, 1), Vertex(1, 1, 1)])
        tri5 = Tetrahedron([Vertex(0, 1, 0), Vertex(1, 0, 0), Vertex(1, 1, 1), Vertex(1, 1, 0)])

        return [tri1, tri2, tri3, tri4, tri5]


class QuarTet:

    def __init__(self, depth):
        # We start with 3D grid NxNxN and devide each child-cube to 5 tetrahedrons
        # unit cube:

        curr_tetrahedrons = UnitCube().divide()
        for _ in range(depth):
            tmp_curr_tetrahedrons = []
            for tet in curr_tetrahedrons:
                tmp_curr_tetrahedrons += tet.sub_divide()
            for tet in tmp_curr_tetrahedrons:
                tet.update_after_all_finish_sub_divide()
            curr_tetrahedrons = tmp_curr_tetrahedrons

    def update_by_deltas(self, vertices_deltas):
        pass

    def init_occupancy_with_SDF(self, SDF):
        # TODO: that will improve results
        pass
