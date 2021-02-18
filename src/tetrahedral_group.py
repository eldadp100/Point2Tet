"""

    Data structure to hold tetrahedrons that all *shares the same vertex* and *exists \epsilon > 0 s.t the union includes a sphere with radios \epsilon*.

    Each vertex has this data-structure as a field. we need - efficient maintaining + efficient inside/outside query and outside distance to define a hinge loss.

"""
import time


class TetsGroupSharesVertex:
    time_monitor = {
        "init": [0., 0.],
        "query_direction": [0., 0.]
    }

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

        def get_tet():
            for tet in self.tets_list:
                curr_side = None
                is_in = True
                for face in tet.faces_by_vertex[self.v]:
                    face_plane = face.plane
                    side = face_plane.get_point_side(self.v.loc + direction)
                    if curr_side is None:
                        curr_side = side
                    else:
                        if side != curr_side:
                            is_in = False
                            break
                if is_in:
                    return tet, curr_side

        tet, side = get_tet()
        assert tet is not None

        TetsGroupSharesVertex.time_monitor["query_direction"][0] += 1
        TetsGroupSharesVertex.time_monitor["query_direction"][1] += time.time() - start_time

        return tet.faces_by_vertex_opposite[self.v].plane.signed_distance(self.v.loc + direction, side)
