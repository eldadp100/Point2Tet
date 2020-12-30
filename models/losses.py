import torch
from pytorch3d.loss import chamfer_distance


def chamfer_distance_quartet_to_point_cloud(quartet, pc, quartet_N_points=3000):
    quartet_pc = quartet.sample_point_cloud(quartet_N_points)
    return chamfer_distance(quartet_pc.unqueeze(1), pc.unqueeze(1))

