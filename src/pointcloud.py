import torch
import _utils
import numpy as np


def face_ray_intersect(ray, face):
    """

    :param ray: (point, direction)
    :param face: the 3 defining points (p_1, p_2, p_3)
    :return: True iff the ray intersect the face

    How?
    1. calculate the plain corresponds to the face
    2. calculate intersection point with the ray
    3. check whether the intersection point inside the face by checking for adjacent pairs in clockwise order if the
       point is from the same size (or more simply right side..).
    """



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
    """


class PointCloud:
    def __init__(self):
        self.points = None

    def init_with_points(self, points):
        self.points = torch.tensor(points)

    def load_file(self, path):
        # .obj support only
        xyz = []
        with open(path, 'r') as f:
            for line in f:
                xyz.append([float(v) for v in line.split()[1:]])

        self.points = torch.tensor(xyz)

    def fill_iterior_of_point_cloud(self, N=9, drop=0.1, method='steps'):

        if method == 'using mesh':
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
            # face_to_plane = {}
            # for face in mesh.faces:
            #     mat = np.ones(3, 4)
            #     mat[:, :3] = np.array(face)
            #     solution = np.linalg.solve(mat, np.zeros(3))
            #     face_to_plane[face] = solution
            #
            #
            # """
            # count number of intersections:
            #     for f in faces if intersect(f, ray)
            # """
            # for point in sampled_points:
            #     num_intersections
            #     for f in mesh.faces:



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

    def normalize(self):
        self.points -= self.points.permute(1, 0).mean(dim=1)
        self.points /= 2 * self.points.permute(1, 0).abs().max(dim=1).values
        self.points += 0.5

    def write_to_file(self, filename):
        with open(filename, "w") as output_file:
            for x, y, z in self.points:
                output_file.write(f"v {x} {y} {z}\n")

        # calculate SDF
        # sample from the whole cube N points and filter positive SDF points (leave only negative SDF)
        # replace the point cloud with the points we sampled
        # add N points from the boundary (some of the original points)
        # we get 2N points

    def __iter__(self):
        return self.points.__iter__()


if __name__ == "__main__":
    device = 'cpu'
    input_xyz, input_normals = _utils.read_pts("../pc.ply")
    input_xyz = torch.Tensor(input_xyz).type(torch.FloatTensor).to(device)[None, :,
                :]  # .type() also changes device somewhy on the server
    input_normals = torch.Tensor(input_normals).type(torch.FloatTensor).to(device)[None, :, :]
    input_xyz, input_normals = input_xyz.squeeze(0), input_normals.squeeze(0)

    # normalize point cloud to [0,1]^3 (Unit Cube)
    input_xyz -= input_xyz.permute(1, 0).mean(dim=1)
    # input_xyz /= 2 * input_xyz.permute(1, 0).max(dim=1).values
    # input_normals /= 2 * input_xyz.permute(1, 0).max(dim=1).values
    # input_xyz += 0.5
    # input_normals += 0.5

    pc = PointCloud()
    pc.init_with_points(input_xyz)
    pc.fill_iterior_of_point_cloud(15)
    pc.write_to_file("filled_pc2.obj")
