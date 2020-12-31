import numpy as np
import torch
from structures.QuarTet import QuarTet
from structures.PointCloud import PointCloud
from utils.visualizer import visualize_quartet, visualize_pointcloud
from models import losses

def main():
    # pc = PointCloud(np.random.rand(100, 3))
    # visualize_pointcloud(pc)
    # visualize_quartet([QuarTet(1)])

    # src_pc = torch.randn(1, 100, 3)#.cuda()
    # dst_pc = torch.randn(1, 50, 3)#.cuda()

    quartet = QuarTet(1)
    pc = PointCloud(np.random.rand(3000, 3))
    print(losses.chamfer_dist_quartet_to_pc(quartet, pc))


if __name__ == "__main__":
    main()