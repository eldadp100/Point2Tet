"""

    Data structure to hold tetrahedrons that all *shares the same vertex* and *exists \epsilon > 0 s.t the union includes a sphere with radios \epsilon*.

    Each vertex has this data-structure as a field. we need - efficient maintaining + efficient inside/outside query and outside distance to define a hinge loss.

"""
import time
import torch


class TetsGroupSharesVertex:
    time_monitor = {
        "init": [0., 0.],
        "query_direction": [0., 0.]
    }
    counter = 0

    def __init__(self, v, tets_list):
        """
        :param v: the vertex in question
        :param tets_list: the tetrahedrons that the vertex `v` is part of
        """
        start_time = time.time()
        self.tets_list = tets_list
        self.v = v
        TetsGroupSharesVertex.update_time_monitor(start_time, "init")
        self.radios = None

    @staticmethod
    def update_time_monitor(start_time, key):
        TetsGroupSharesVertex.time_monitor[key][0] += 1
        TetsGroupSharesVertex.time_monitor[key][1] += time.time() - start_time

    def signed_distance_to_face_in_direction(self, direction):
        """
        :param direction: direction vector pointing to the direction in which we look for the face
        :return: a tuple containing the distance of the opposite face to the `direction` vector from the original vertex
            and the query point (the original vertex plus the direction vector) in this order, or None if outside the
            tetrahedrons containing the vertex
        """
        start_time = time.time()
        query_point = self.v.original_loc + direction

        def get_tet():
            """ :return: the tetrahedron from self.tets_list containing the query point, or None if none exists """
            for tet in self.tets_list:
                tet_hfs = tet.faces_by_vertex[self.v.get_original_xyz()]
                sides = [hf.plane.get_point_side(query_point) for hf in tet_hfs]
                if not (False in sides):
                    return tet
            return None

        tet = get_tet()
        if tet is not None:
            TetsGroupSharesVertex.update_time_monitor(start_time, "query_direction")
            opposite_face_plane = tet.faces_by_vertex_opposite[self.v.get_original_xyz()].plane
            orig_dis = opposite_face_plane.signed_distance(self.v.original_loc)
            curr_dis = opposite_face_plane.signed_distance(query_point)
            return max(curr_dis, orig_dis / 2) - orig_dis / 2
        else:
            TetsGroupSharesVertex.counter += 1
            if TetsGroupSharesVertex.counter % 100 == 0:
                print(TetsGroupSharesVertex.counter)

            return None

    def update_tets_list(self, to_remove, to_add):
        """
        :param to_remove: an iterable containing tetrahedrons to remove
        :param to_add: an iterable containing tetrahedrons to add
        """
        for _to_remove in to_remove:
            self.tets_list.remove(_to_remove)
        for _to_add in to_add:
            self.tets_list.append(_to_add)
        self.calculate_max_inner_sphere()

    def calculate_max_inner_sphere(self):
        self.radios = min(
            [-tet.faces_by_vertex_opposite[self.v.get_original_xyz()].plane.signed_distance(self.v.original_loc) for tet
             in self.tets_list])
        assert self.radios >= 0, "FUCK"

    def inner_sphere_loss(self, move_direction):
        if self.radios is None:
            self.calculate_max_inner_sphere()
        move_direction_norm = torch.norm(move_direction)
        return max(move_direction_norm, self.radios / 2) - self.radios / 2


if __name__ == '__main__':
    import quartet

    a = quartet.QuarTet('../objects/cube_0.15.tet')
    x = a.vertices[50]
