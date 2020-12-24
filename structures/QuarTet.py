import numpy as np
import torch


def tensors_eq(v1, v2):
    return len(v1) == len(v2) and (v1 == v2).sum() == len(v2)


class Tetrahedron:
    def __init__(self, vertices, depth=0):
        self.vertices = vertices
        self.occupancy = np.random.choice([0, 1])  # very small chance to all be 0
        self.neighborhood = set()
        self.features = torch.stack(self.vertices).permute(1, 0).sum() / 4
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
                if tensors_eq(v1, v2):
                    c += 1
        return c == 3

    def get_center(self):
        a = torch.stack([v for v in self.vertices])
        loc = a.permute(1, 0).sum(dim=1) / 4
        return loc

    def sub_divide(self):
        if self.sub_divided is None:
            ret_tets = []
            center = self.get_center()
            for remove_vertex in self.vertices:
                new_tet = []
                for v in self.vertices:
                    if not tensors_eq(v, remove_vertex):
                        new_tet.append(v)
                new_tet.append(center)
                new_tet = Tetrahedron(new_tet, depth=self.depth + 1)
                ret_tets.append(new_tet)

            self.sub_divided = ret_tets

        return self.sub_divided

    def __hash__(self):
        return (self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]).__hash__()

    def update_by_deltas(self, vertices_deltas):
        new_vertices = []
        for i, v in enumerate(self.vertices):
            new_vertices.append(v + vertices_deltas[i])
        self.vertices = new_vertices

    @staticmethod
    def determinant(mat):
        a = mat[0][0]
        b = mat[0][1]
        c = mat[0][2]
        d = mat[1][0]
        e = mat[1][1]
        f = mat[1][2]
        g = mat[2][0]
        h = mat[2][1]
        i = mat[2][2]

        return (
                a * e * i +
                b * f * g +
                c * d * h -
                c * e * g -
                b * d * i -
                a * f * h
        )

    def calculate_volume(self):
        return Tetrahedron.determinant(self.vertices) / 6


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
            if tensors_eq(v1, v2):
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


def Vertex(x, y, z):
    return torch.tensor([x, y, z])


class UnitCube:
    def __init__(self):
        pass

    def divide(self):
        tri1 = Tetrahedron([Vertex(0, 1, 0), Vertex(0, 1, 1), Vertex(1, 1, 1), Vertex(0, 0, 1)])
        tri2 = Tetrahedron([Vertex(0, 1, 0), Vertex(1, 1, 1), Vertex(0, 0, 1), Vertex(1, 0, 0)])
        tri3 = Tetrahedron([Vertex(0, 0, 0), Vertex(1, 0, 0), Vertex(0, 0, 1), Vertex(0, 1, 0)])
        tri4 = Tetrahedron([Vertex(1, 0, 0), Vertex(1, 0, 1), Vertex(0, 0, 1), Vertex(1, 1, 1)])
        tri5 = Tetrahedron([Vertex(0, 1, 0), Vertex(1, 0, 0), Vertex(1, 1, 1), Vertex(1, 1, 0)])

        return [tri1, tri2, tri3, tri4, tri5]


class QuarTet:
    def __init__(self, depth):
        # We start with 3D grid NxNxN and devide each child-cube to 5 tetrahedrons
        # unit cube:
        self.curr_tetrahedrons = UnitCube().divide()
        for _ in range(depth):
            tmp_curr_tetrahedrons = []
            for tet in self.curr_tetrahedrons:
                tmp_curr_tetrahedrons += tet.sub_divide()

            self.curr_tetrahedrons = tmp_curr_tetrahedrons
        calculate_and_update_neighborhood(self.curr_tetrahedrons)
        self.fill_neighbors()

    def fill_neighbors(self):
        for tet in self.curr_tetrahedrons:
            for i in range(4 - len(tet.neighborhood)):
                tet.add_neighbor(tet)

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
            visited_tets.add(tet)
            visited_tets.add(neighbor)
            faces.append(Face(tet, neighbor))

        return faces

    def __iter__(self):
        return self.curr_tetrahedrons.__iter__()

    def get_occupied_tets(self):
        result = []
        for tet in self:
            if tet.occupancy == 1:
                result.append(tet)
        return result

    def sample_point_cloud(self, pc_size):
        samples = []
        occupied_tets = self.get_occupied_tets()
        volumes = [tet.volume() for tet in occupied_tets]
        volumes_total = sum(volumes)

        points_count = [int(np.round((volume / volumes_total) * pc_size)) for volume in volumes]

        for i, tet in enumerate(occupied_tets):
            for _ in points_count[i]:
                samples.append(sum([vertex * np.random.uniform(0, 1) for vertex in tet.vertices]))


if __name__ == '__main__':
    a = QuarTet(2)
    for tet in a:
        print(len(tet.neighborhood))
    b = a.sample_disjoint_faces(4)
    c = a.get_occupied_tets()
    print(b)
    print(c)
