import numpy as np
import torch
from structures.QuarTet import QuarTet
from structures.PointCloud import PointCloud
from utils.visualizer import visualize_quartet, visualize_pointcloud

def main():
    # pc = PointCloud(np.random.rand(100, 3))
    # visualize_pointcloud(pc)

    visualize_quartet([QuarTet(1)])


if __name__ == "__main__":
    main()