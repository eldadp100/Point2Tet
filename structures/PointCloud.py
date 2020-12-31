import torch

class PointCloud:
    def __init__(self, points):
        self.points = torch.tensor(points)

    def __iter__(self):
        return self.points.__iter__()

    def fill_iterior_of_point_cloud(self, N=9):
        """
            N: dense measurement
        """
        multiple = float(1) / (N + 1)
        curr_mult = 1 - multiple
        original_points = self.points
        for i in range(N):
            new_points = original_points.copy()
            self.points.extend(new_points * curr_mult)
            curr_mult -= multiple


        # calculate SDF
        # sample from the whole cube N points and filter positive SDF points (leave only negative SDF)
        # replace the point cloud with the points we sampled
        # add N points from the boundary (some of the original points)
        # we get 2N points
