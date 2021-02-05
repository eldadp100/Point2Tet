import argparse
import torch
import pointcloud
import mesh
import numpy as np


def face_ray_intersect(ray, face, face_plane):
    """

    :param ray: (point, direction)
    :param face: the 3 defining points (p_1, p_2, p_3)
    :param face_plane: plane to save computation cost
    :return: True iff the ray intersect the face

    How?
    1. calculate the plain corresponds to the face
    2. calculate intersection point with the ray
    3. check whether the intersection point inside the face by checking for adjacent pairs in clockwise order if the
       point is from the same size (or more simply right side..).
    """

    a, b, c, d = face_plane  # the plane
    """
        calculate intersection point - search for \lambda that satisfy:

        (a, b, c)^T (start_point + lambda * direction) + d = 0
        --> lambda = (-d - (a, b, c)^T (start_point)) / ((a, b, c)^T direction)

        Then if lambda < 0 it's in the oposite side of the ray and we return False
    """
    _lambda = (-d - np.dot(np.array([a, b, c]), np.array(ray[0]))) / (np.dot(np.array([a, b, c]), ray[1]))
    if _lambda < 0:
        return False
    the_intersection_point = ray[0] + _lambda * ray[1]
    """
            check if the intersecting point in the face 
            by projecting the points on xy and check there if is inside
    """
    if face[0][0] == face[1][0] == face[2][0] or face[0][1] == face[1][1] == face[2][1]:
        """project on zy"""
        curr = None
        for i in range(3):
            j = (i + 1) % 3
            if face[i][2] == face[j][2]:
                i_j_line = [1, 0, face[i][1] - face[j][1]]
            else:
                m = (face[i][1] - face[j][1]) / (face[i][2] - face[j][2])
                i_j_line = [m, -1, face[i][1] - m * face[i][2]]
            if curr is None:
                curr = np.dot(np.array(i_j_line),
                              np.array([the_intersection_point[2], the_intersection_point[1], 1.])) > 0
            else:
                new = np.dot(np.array(i_j_line),
                             np.array([the_intersection_point[2], the_intersection_point[1], 1.])) > 0
                if curr != new:
                    return False

    else:
        """project on xy"""
        curr = None
        for i in range(3):
            j = (i + 1) % 3
            if face[i][0] == face[j][0]:
                i_j_line = [1, 0, face[i][1] - face[j][1]]
            else:
                m = (face[i][1] - face[j][1]) / (face[i][0] - face[j][0])
                i_j_line = [m, -1, face[i][1] - m * face[i][0]]
            if curr is None:
                curr = np.dot(np.array(i_j_line),
                              np.array([the_intersection_point[0], the_intersection_point[1], 1.])) > 0
            else:
                new = np.dot(np.array(i_j_line),
                             np.array([the_intersection_point[0], the_intersection_point[1], 1.])) > 0
                if curr != new:
                    return False


class FillPointCloud(pointcloud.PointCloud):

    def fill_interior_of_point_cloud(self, method='steps', _mesh=None, N=9, drop=0.1):

        if method == 'mesh':
            """
                1. sample points uniformly in the enclosing cube
                2. for each point pass a ray in any direction (e.g. x direction...) and count number of intersections.
                        If even -> outside the mesh
                        If odd  -> inside the mesh
                    we drop all points that are outside the mesh
                3. return the remaining points.                
            """

            """
                Face to plane dict.
                calculate the plane = solve the following linear system
                    a x_1 + b y_1 + c y_1 + d = 0
                    a x_2 + b y_2 + c y_2 + d = 0
                    a x_3 + b y_3 + c y_3 + d = 0
            """
            face_to_plane = {}
            for face in _mesh.faces:
                mat = np.ones((4, 4))
                mat[:3, :3] = np.array(face)
                mat[3, 3] = 0
                b = np.zeros(4)
                b[3] = -1
                solution = np.linalg.solve(mat, b)
                face_to_plane[tuple(map(tuple, face))] = solution

            """
                sample and filter points
            """
            sampled_points = np.random.rand(10000, 3)
            filtered_point = []
            for point in sampled_points:
                direction = np.array([0., 0., 1.])
                num_intersections = 0
                for f in _mesh.faces:
                    if face_ray_intersect((point, direction), f, face_to_plane[tuple(map(tuple, f))]):
                        num_intersections += 1

                if num_intersections % 2 == 0:
                    filtered_point.append(point)

            self.points = filtered_point

        elif method == 'winding number':
            """
            """
            raise Exception("not implemented")

        elif method == 'SDF':
            """
                1. calculate SDF of the points (maybe with SIREN)
                2. draw points uniformly and remove those with positive SDF value
                3. return the remaining points.
            """
            raise Exception("not implemented")

        elif method == 'steps':
            multiple = float(1) / (N + 1)
            curr_mult = 1 - multiple
            original_points = self.points
            for i in range(N):
                # self.points = torch.cat([self.points, torch.tensor([1, curr_mult, 1]) * original_points])
                new_points = curr_mult * original_points
                size = new_points.shape[0]
                indices = np.random.randint(0, size, int(drop * size))
                new_points = new_points[indices]
                self.points = torch.cat([self.points, new_points])
                curr_mult -= multiple

        else:
            raise Exception("not valid fill interior method")


# argument parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Filling Arguments')
    parser.add_argument('--input_pc', type=str, default='../filled_sphere.obj', help='.obj format')
    parser.add_argument('--input_mesh', type=str, default=None, help='optionality!!! .obj format')
    parser.add_argument('--output_path', type=str, default='default_name', help='output path')
    parser.add_argument('--method', type=str, default='default_name', help='steps (for convex) / mesh / winding number')
    opts = parser.parse_args()
    device = 'cpu'

if __name__ == "__main__":
    pc = FillPointCloud()
    pc.load_file(opts.input_pc)
    _mesh = None
    if opts.input_mesh is not None:
        _mesh = mesh.Mesh(opts.input_mesh)

    pc.fill_interior_of_point_cloud(method=opts.method, _mesh=_mesh)
    pc.write_to_file(opts.output_path)
