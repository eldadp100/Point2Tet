import torch
import _utils
import numpy as np
import mesh
from mesh_to_sdf.surface_point_cloud import SurfacePointCloud
import pyrender
import trimesh
from scipy.spatial import ConvexHull

from _utils import read_pts


class PointCloud:
    def __init__(self):
        self.points = None
        self.normals = None

    def init_with_points(self, points):
        self.points = points.clone().detach()

    def load_with_normals(self, path):
        points, normals = read_pts(path)
        self.points = torch.tensor(points)
        self.normals = torch.tensor(normals)

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
        # self.points -= self.points.min()
        # Z = self.points.permute(1, 0).max(dim=1).values
        # self.points /= Z
        # self.normals /= Z

        self.points -= self.points.permute(1, 0).mean(dim=1)
        self.points /= 2 * self.points.permute(1, 0).abs().max(dim=1).values
        self.points += 0.5
        self.points /= 2


    def write_to_file(self, filename):
        with open(filename, "w") as output_file:
            for x, y, z in self.points:
                output_file.write(f"v {x} {y} {z}\n")

        # calculate SDF
        # sample from the whole cube N points and filter positive SDF points (leave only negative SDF)
        # replace the point cloud with the points we sampled
        # add N points from the boundary (some of the original points)
        # we get 2N points

    def calc_sdf(self, query_points):
        spc = SurfacePointCloud(
            None,
            points=self.points,
            normals=self.normals
        )

        return spc.get_sdf_in_batches(query_points)

    def visualize_sdf(self, query_points, sdf):
        colors = np.zeros(query_points.shape)
        colors[sdf < 0, 2] = 1
        colors[sdf > 0, 0] = 1
        cloud = pyrender.Mesh.from_points(query_points, colors=colors)
        scene = pyrender.Scene()
        scene.add(cloud)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

    def __iter__(self):
        return self.points.__iter__()

    def convex_hull(self, filename=None, return_trimesh=False):
        hull = ConvexHull(np.array(self.points))
        if filename is not None:
            points = []
            for point in hull.points:
                x, y, z = point
                points.append(f"v {x} {y} {z}")

            faces = []
            for face in hull.neighbors:
                x, y, z = face
                faces.append(f"f {x} {y} {z}")

            with open(filename, "w") as file:
                file.write('\n'.join(points))
                file.write('\n')
                file.write('\n'.join(faces))

        if return_trimesh:
            return trimesh.Trimesh(vertices=hull.points, faces=hull.simplices)
        else:
            return hull


if __name__ == "__main__":
    # device = 'cpu'
    # input_xyz, input_normals = _utils.read_pts("../objects/pc.ply")
    # input_xyz = torch.Tensor(input_xyz).type(torch.FloatTensor).to(device)[None, :,
    #             :]  # .type() also changes device somewhy on the server
    # input_normals = torch.Tensor(input_normals).type(torch.FloatTensor).to(device)[None, :, :]
    # input_xyz, input_normals = input_xyz.squeeze(0), input_normals.squeeze(0)

    # normalize point cloud to [0,1]^3 (Unit Cube)
    # input_xyz -= input_xyz.permute(1, 0).mean(dim=1)
    # input_xyz /= 2 * input_xyz.permute(1, 0).max(dim=1).values
    # input_normals /= 2 * input_xyz.permute(1, 0).max(dim=1).values
    # input_xyz += 0.5
    # input_normals += 0.5

    # pc = PointCloud()
    # pc.init_with_points(input_xyz)
    # pc.fill_iterior_of_point_cloud(method='steps')
    # pc.write_to_file("pc.obj")
    #
    # _mesh = mesh.Mesh('../objects/init_mesh.obj')
    # pc = PointCloud()
    # pc.load_file('../pc.obj')
    # pc.fill_iterior_of_point_cloud(method='mesh', mesh=_mesh)
    # pc.write_to_file("pc.obj")

    pc = PointCloud()
    pc.load_file('../objects/filled_sphere.obj')
    xyz = pc.points
    print(f'max points = {xyz.max(axis=0)}')
    print(f'min points = {xyz.min(axis=0)}')
