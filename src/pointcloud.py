import torch
import _utils
import numpy as np
import mesh


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
                if line == '\n':
                    continue
                split = line.split()
                if split[0] == 'v':
                    xyz.append([float(v) for v in line.split()[1:]])

        self.points = torch.tensor(xyz)

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
    input_xyz, input_normals = _utils.read_pts("../objects/pc.ply")
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

    # pc = PointCloud()
    # pc.init_with_points(input_xyz)
    # pc.fill_iterior_of_point_cloud(method='steps')
    # pc.write_to_file("pc.obj")
    #
    _mesh = mesh.Mesh('../objects/init_mesh.obj')
    pc = PointCloud()
    pc.load_file('../pc.obj')
    pc.fill_iterior_of_point_cloud(method='mesh', mesh=_mesh)
    pc.write_to_file("pc.obj")
