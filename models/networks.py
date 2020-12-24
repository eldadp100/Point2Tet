import torch
import torch.nn as nn
from torch import optim
import numpy as np

from structures.QuarTet import QuarTet, Tetrahedron

NEIGHBORHOOD_SIZE = 5


class MotherCubeConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(MotherCubeConv, self).__init__()
        self.lin = nn.Linear(NEIGHBORHOOD_SIZE * in_features, out_features)


    def forward(self, mother_cube: QuarTet):
        for tet in mother_cube:
            neighborhood_features = [tet.features]
            for n_tet in tet.neighborhood:
                neighborhood_features.append(n_tet.prev_features)
            neighborhood_features = torch.cat(neighborhood_features, dim=0)
            tet.features = self.lin(neighborhood_features)
        for tet in mother_cube:
            tet.prev_features = tet.features


class MotherCubePool(nn.Module):
    def __init__(self, pool_target):
        super(MotherCubePool, self).__init__()
        self.pool_target = pool_target

    def pool(self, mother_cube, pool_target):
        sampled_faces = mother_cube.sample_disjoint_faces(pool_target)
        for sampled_face in sampled_faces:
            tet1, tet2 = sampled_face.get_tets()
            shared_features = torch.max(torch.cat([tet1, tet2], dim=1), dim=-1).detach()
            tet1.features = shared_features.clone()
            tet2.features = shared_features.clone()

    def unpool(self):
        pass


class TetCNN_PP(nn.Module):
    def __init__(self, ncf):
        super(TetCNN_PP, self).__init__()
        self.ncf = ncf
        for i, num_features in enumerate(ncf[:-1]):
            setattr(self, f'conv{i}', MotherCubeConv(num_features, ncf[i + 1]))
            # setattr(self, f'pool{i}', MotherCubePool(num_features, ))

    def forward(self, mother_cube):
        for i in range(len(self.ncf) - 1):
            getattr(self, f'conv{i}')(mother_cube)



class OurNet(nn.Module):
    def __init__(self):
        super(OurNet, self).__init__()

        ncf = [3, 32, 64, 64]

        self.conv_net = TetCNN_PP(ncf)  # TetCNN++
        self.net_vertices_movements = nn.Linear(64, 12)  # 3D movement
        self.net_occupancy = nn.Linear(64, 1)  # Binary classifier - occupancy

    def forward(self, mother_cube):
        self.conv_net(mother_cube)
        for tet in mother_cube:
            tet_deltas = self.net_vertices_movements(tet.features).reshape(4, 3)
            tet.update_by_deltas(tet_deltas)
            tet.occupancy = self.net_occupancy(tet.features).item()


def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def get_scheduler(iters, optim):
    lr_lambda = lambda x: 1 - min((0.1 * x / float(iters), 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    return scheduler


def init_net(opts, device):
    net = OurNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=opts.lr)
    scheduler = get_scheduler(opts.iterations, optimizer)

    return net, optimizer, scheduler


def main():
    MotherCube = QuarTet(1)
    net = OurNet()
    print(net)
    print(net(MotherCube))


if __name__ == "__main__":
    main()