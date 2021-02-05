import argparse


class Options:
    def __init__(self):
        self.args = None
        self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Point2Mesh options')
        parser.add_argument('--name', type=str, default='./checkpoints/guitar', help='path to save results to')
        parser.add_argument('--input-pc', type=str, default='filled_pc2.ply', help='input point cloud')
        parser.add_argument('--continue_train', type=bool, default=False, help='continue train from the latest model')
        parser.add_argument('--chamfer_sample_size', type=int, default=1000,
                            help='number of points to sample for chamfer loss')
        parser.add_argument('--sample_size', type=int, default=1000, help='number of points to sample from the pc')
        parser.add_argument('--init_quartet_size', type=int, default=2, help='number of unit cubes in the initial quartet')

        # HYPER PARAMETERS - RECONSTRUCTION
        parser.add_argument('--torch-seed', type=int, metavar='N', default=5, help='torch random seed')
        parser.add_argument('--for_chamfer_samples', type=int, metavar='N', default=25000)
        parser.add_argument('--iterations', type=int, metavar='N', default=10000, help='number of iterations to do')
        parser.add_argument('--upsamp', type=int, metavar='N', default=1000, help='upsample each {upsamp} iteration')

        # HYPER PARAMETERS - NETWORK
        parser.add_argument('--lr', type=float, metavar='1eN', default=1.1e-4, help='learning rate')
        parser.add_argument('--res-blocks', type=int, metavar='N', default=3, help='')
        parser.add_argument('--ncf', nargs='+', default=[16, 32, 64, 64, 128], type=int, help='convs to do')
        parser.add_argument('--pr', nargs='+', default=[0.3] * 5, type=float, help='pooling ratios to do')
        parser.add_argument('--save_freq', type=int, default=100, help='save the model each {save_freq} iterations')

        # MULTI GPUS TRAINING

        self.args = parser.parse_args()
