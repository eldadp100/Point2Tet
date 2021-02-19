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
        start_time = time.time()
        self.tets_list = tets_list
        self.v = v

        TetsGroupSharesVertex.time_monitor["init"][0] += 1
        TetsGroupSharesVertex.time_monitor["init"][1] += time.time() - start_time

    def query_direction(self, direction):
        """
        return signed distance to the face opposite to the vertex moved in the direction
        """
        start_time = time.time()
        query_point = self.v.original_loc + direction
        def get_tet():
            for tet in self.tets_list:
                tet_hfs = tet.faces_by_vertex[self.v.get_original_xyz()]
                sides = [hf.plane.get_point_side(query_point) for hf in tet_hfs]
                if not (False in sides):
                    return tet
            return None

        tet = get_tet()
        if tet is not None:
            TetsGroupSharesVertex.time_monitor["query_direction"][0] += 1
            TetsGroupSharesVertex.time_monitor["query_direction"][1] += time.time() - start_time
            return tet.faces_by_vertex_opposite[self.v.get_original_xyz()].plane.signed_distance(
                self.v.original_loc + direction)
        else:
            TetsGroupSharesVertex.counter += 1
            if TetsGroupSharesVertex.counter % 100 == 0:
                print(TetsGroupSharesVertex.counter)

            return torch.tensor(0.)
