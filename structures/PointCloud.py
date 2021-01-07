import torch
import point2tet_utils
import numpy as np


class PointCloud:
    def __init__(self, points):
        self.points = torch.tensor(points)

    def __iter__(self):
        return self.points.__iter__()

    def fill_iterior_of_point_cloud(self, N=9, drop=0.1):
        """
            N: dense measurement
        """
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


def main():
    device = 'cpu'
    input_xyz, input_normals = point2tet_utils.read_pts("../pc.ply")
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

    pc = PointCloud(input_xyz)
    pc.fill_iterior_of_point_cloud(15)
    pc.write_to_file("filled_pc2.obj")


if __name__ == "__main__":
    main()