class PointCloud:

    def fill_iterior_of_point_cloud(self, N):
        """
            N: dense measurement
        """

        # calculate SDF
        # sample from the whole cube N points and filter positive SDF points (leave only negative SDF)
        # replace the point cloud with the points we sampled
        # add N points from the boundary (some of the original points)
        # we get 2N points
