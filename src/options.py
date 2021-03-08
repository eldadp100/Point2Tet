import argparse


class Options:
    def __init__(self):
        self.args = None
        self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Point2Mesh options')
        parser.add_argument('--name', type=str, default='default_name',
                            help='name in checkpoint folder to save results to')
        parser.add_argument('--continue_train', action='store_true', help='continue train from latest checkpoint')
        parser.add_argument('--save_freq', type=int, default=100, help='save the model each {save_freq} iterations')
        parser.add_argument('--checkpoint_folder', type=str, default="../checkpoints", help='')
        parser.add_argument('--init_cube', type=str, default="../objects/cube_0.1.tet", help='.tet format')
        parser.add_argument('--input_filled_pc', type=str, default='../objects/filled_sphere.obj', help='.obj format')
        parser.add_argument('--input_pc', type=str, default='./filled_sphere.obj',
                            help='input point cloud with normals in .ply format')
        parser.add_argument('--iteration_number', type=int, default=-1,
                            help='the number of iteration to continue from in case of using --continue_train')
        parser.add_argument('--cache_folder', type=str, default='../quartet_cache',
                            help='a folder containing initial quartet objects for faster loading times')

        # HYPER PARAMETERS - RECONSTRUCTION
        parser.add_argument('--torch-seed', type=int, metavar='N', default=5, help='torch random seed')
        parser.add_argument('--chamfer_samples', type=int, metavar='N', default=1000, help='resampled every iteration')
        parser.add_argument('--iterations', type=int, metavar='N', default=100000, help='number of iterations to do')
        parser.add_argument('--upsamp', type=int, metavar='N', default=1000, help='upsample each {upsamp} iteration')

        # HYPER PARAMETERS - NETWORK
        parser.add_argument('--lr', type=float, metavar='1eN', default=1.1e-4, help='learning rate')
        parser.add_argument('--res-blocks', type=int, metavar='N', default=3, help='')
        parser.add_argument('--ncf', nargs='+', default=[30, 64, 64, 64], type=int, help='convs to do')
        parser.add_argument('--pr', nargs='+', default=[0.3] * 5, type=float, help='pooling ratios to do')

        self.args = parser.parse_args()
