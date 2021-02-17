import argparse
import time

import torch
import pointcloud
import mesh
import numpy as np


# def face_ray_intersect(ray, face, face_plane):
#     """
#
#     :param ray: (point, direction)
#     :param face: the 3 defining points (p_1, p_2, p_3)
#     :param face_plane: plane to save computation cost
#     :return: True iff the ray intersect the face
#
#     How?
#     1. calculate the plain corresponds to the face
#     2. calculate intersection point with the ray
#     3. check whether the intersection point inside the face by checking for adjacent pairs in clockwise order if the
#        point is from the same size (or more simply right side..).
#     """
#
#     a, b, c, d = face_plane  # the plane
#     """
#         calculate intersection point - search for \lambda that satisfy:
#
#         (a, b, c)^T (start_point + lambda * direction) + d = 0
#         --> lambda = (-d - (a, b, c)^T (start_point)) / ((a, b, c)^T direction)
#
#         Then if lambda < 0 it's in the oposite side of the ray and we return False
#     """
#     _lambda = (-torch.matmul(ray[0], torch.tensor([a, b, c])) - d / torch.dot(torch.tensor(ray[1], dtype=torch.float64),
#                                                                               torch.tensor([a, b, c])))
#     falsers = _lambda < 0
#
#     the_intersection_points = _lambda.expand(3, _lambda.shape[0]).permute(1, 0) * ray[1].expand(_lambda.shape[0], 3)
#     """
#             check if the intersecting point in the face
#             by projecting the points on xy and check there if is inside
#     """
#     if face[0][0] == face[1][0] == face[2][0] or face[0][1] == face[1][1] == face[2][1]:
#         """project on zy"""
#         curr = None
#         for i in range(3):
#             j = (i + 1) % 3
#             if face[i][2] == face[j][2]:
#                 i_j_line = [1, 0, face[i][1] - face[j][1]]
#             else:
#                 m = (face[i][1] - face[j][1]) / (face[i][2] - face[j][2])
#                 i_j_line = [m, -1, face[i][1] - m * face[i][2]]
#
#             new = torch.matmul(the_intersection_points[:, 1:3],
#                                torch.tensor([i_j_line[0], i_j_line[1]]).type_as(the_intersection_points))
#             new = (new + i_j_line[2]) > 0
#             if curr is None:
#                 curr = new
#             else:
#                 curr = curr * new
#
#     else:
#         """project on xy"""
#         curr = None
#         for i in range(3):
#             j = (i + 1) % 3
#             if face[i][0] == face[j][0]:
#                 i_j_line = [1, 0, face[i][1] - face[j][1]]
#             else:
#                 m = (face[i][1] - face[j][1]) / (face[i][0] - face[j][0])
#                 i_j_line = [m, -1, face[i][1] - m * face[i][0]]
#
#             new = torch.matmul(the_intersection_points[:, 0:2],
#                                torch.tensor([i_j_line[0], i_j_line[1]]).type_as(the_intersection_points))
#             new = (new + i_j_line[2]) > 0
#             if curr is None:
#                 curr = new
#             else:
#                 curr = curr * new
#
#         return curr * falsers



def face_ray_intersect(rays, face, face_plane):
    intersections = []
    result = []
    a, b, c, d = face_plane
    ray_direction = rays[1]
    for l0 in rays[0]:
        normal = torch.tensor([a, b, c])
        p0 = face[0]
        # l0, ray_direction = ray
        denominator = torch.matmul(ray_direction, normal)
        nominator = torch.matmul((p0 - l0), normal)
        if nominator == 0 and denominator == 0:
            intersections.append(p0)
            result.append(True)
        if denominator != 0:
            d = nominator / denominator
            if d < 0:
                intersections.append(None)
                result.append(False)
            else:
                intersections.append(l0 + d * ray_direction)
                result.append(True)
        else:
            intersections.append(None)
            result.append(False)
    return result, intersections


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

            sampled_points = torch.rand(N, 3, dtype=torch.float64)
            direction = torch.tensor([0., 0., 1.])
            number_intersections = torch.zeros(N)
            for f in _mesh.faces:
                a = face_ray_intersect((sampled_points, direction), f, face_to_plane[tuple(map(tuple, f))])
                if a is not None:
                    number_intersections += a
            remaining_point_indices = number_intersections % 2 == 0
            filtered_points = sampled_points[remaining_point_indices]

            # """
            #     sample and filter points
            # """
            # sampled_points = np.random.rand(10000, 3)
            # filtered_point = []
            # for point in sampled_points:
            #     s = time.time()
            #     direction = np.array([0., 0., 1.])
            #     num_intersections = 0
            #     for f in _mesh.faces:
            #         if face_ray_intersect((point, direction), f, face_to_plane[tuple(map(tuple, f))]):
            #             num_intersections += 1
            #
            #     if num_intersections % 2 == 0:
            #         filtered_point.append(point)
            #     print(time.time() - s)
            self.points = filtered_points

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



# if __name__ == "__main__":
#     face = torch.tensor([
#         [1., 0.5, 0.],
#         [0., 0., 0.],
#         [0., 0., 1.]
#     ], dtype=torch.float64)
#     ray = (torch.tensor([[0.25, 0., 0.25]], dtype=torch.float64), torch.tensor([0., 1., 0.], dtype=torch.float64))
#     face_plane = torch.tensor([-0.5, 1., 0., 0.], dtype=torch.float64)
#
#     intersect, intersection = face_ray_intersect(ray, face, face_plane)
#     print(intersection)

# # test
# if __name__ == '__main__':
#     _mesh = mesh.Mesh('../objects/init_mesh.obj')
#     pc = FillPointCloud()
#     pc.load_file('../objects/pc.obj')
#     pc.fill_interior_of_point_cloud(method='mesh', _mesh=_mesh, N=10000)
#     pc.write_to_file("pc.obj")

# normalize point cloud
# if __name__ == '__main__':
#     pc = FillPointCloud()
#     pc.load_file('../objects/filled_sphere.obj')
#     pc.normalize()
#     pc.write_to_file("../objects/filled_sphere.obj")

# # argument parsing
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Point Cloud Filling Arguments')
#     parser.add_argument('--input_pc', type=str, default='../objects/filled_sphere.obj', help='.obj format')
#     parser.add_argument('--input_mesh', type=str, default=None, help='optionality!!! .obj format')
#     parser.add_argument('--output_path', type=str, default='default_name', help='output path')
#     parser.add_argument('--method', type=str, default='mesh', help='steps (for convex) / mesh / winding number')
#     parser.add_argument('--mesh_N', type=int, default=10000, help='sampling points for mesh')
#     opts = parser.parse_args()
#     device = 'cpu'
#
# if __name__ == "__main__":
#     pc = FillPointCloud()
#     pc.load_file(opts.input_pc)
#     _mesh = None
#     if opts.input_mesh is not None:
#         _mesh = mesh.Mesh(opts.input_mesh)
#
#     pc.fill_interior_of_point_cloud(method=opts.method, _mesh=_mesh, N=opts.mesh_N)
#     pc.write_to_file(opts.output_path)
