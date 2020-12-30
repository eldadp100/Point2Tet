import torch
from pytorch3d.loss import chamfer_distance
import structures.QuarTet

def chamfer_distance_quartet_to_point_cloud(quartet, pc, quartet_N_points=3000):
    quartet_pc = quartet.sample_point_cloud(quartet_N_points)
    return chamfer_distance(quartet_pc, pc)


def main():
    pc = torch.randn((3000, 3))
    quartet = QuarTet()
    chamfer_distance_quartet_to_point_cloud(quartet, pc)


if __name__ == '__main__':
    main()