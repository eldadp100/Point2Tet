import numpy as np
from structures.QuarTet import QuarTet
from structures.PointCloud import PointCloud
from utils.visualizer import visualize_quartet, visualize_pointcloud

def main():
    quartet = QuarTet(0)
    visualize_quartet(quartet)

    pc = PointCloud(np.random.rand(100, 3))
    visualize_pointcloud(pc)

if __name__ == "__main__":
    main()