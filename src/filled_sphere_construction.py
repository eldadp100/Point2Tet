import numpy as np
import torch
from pointcloud import  PointCloud
if __name__ == '__main__':
    N = 10000
    sphere_points = []
    while len(sphere_points) < N:
        rand_point = np.random.rand(3)
        if np.linalg.norm(rand_point - 0.5) <= 0.5:
            sphere_points.append(rand_point)

    out = PointCloud()
    out.init_with_points(torch.from_numpy(np.array(sphere_points)))
    out.write_to_file('./filled_sphere.obj')