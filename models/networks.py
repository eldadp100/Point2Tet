import torch
import torch.nn as nn
from torch import optim

from structures.QuarTet import QuarTet, Tetrahedron

NEIGHBORHOOD_SIZE = 5


class MotherCubeConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(MotherCubeConv, self).__init__()
        self.conv = nn.Linear(NEIGHBORHOOD_SIZE * in_features, out_features)

    def forward(self, mother_cube: QuarTet):
        for tet in mother_cube:
            neighborhood_features = [tet.features]
            for n_tet in tet.neighborhood:
                neighborhood_features.append(n_tet.features)
            neighborhood_features = torch.cat(neighborhood_features, dim=0)
            tet.features = self.conv(neighborhood_features)


class MotherCubePool(nn.Module):
    def __init__(self):
        super(MotherCubePool, self).__init__()
        pass

    def pool(self):
        pass

    def unpool(self):
        pass


class TetCNN_PP(nn.Module):
    def __init__(self, ncf):
        super(TetCNN_PP, self).__init__()
        for i, num_features in enumerate(ncf[:-1]):
            setattr(self, f'conv{i}', MotherCubeConv(num_features, ncf[i + 1]))
            # setattr(self, f'pool{i}', MotherCubePool(num_features, ))

    def forward(self, mother_cube):
        for i in range(len(self.ncf)):
            mother_cube = getattr(self, f'conv{i}')(mother_cube)



class OurNet(nn.Module):
    def __init__(self):
        super(OurNet, self).__init__()

        ncf = [3, 32, 64, 64]

        self.conv_net = TetCNN_PP(ncf)  # TetCNN++
        self.net_verts = nn.Linear(64, 3)  # 3D movement
        self.net_occupancies = nn.Linear(64, 1)  # Binary classifier - occupancy

    def forward(self, mother_cube):
        self.conv_net(mother_cube)

        verts = mother_cube.get_verts()
        delta_verts = self.net_verts(verts)  # vertices direction
        tets = mother_cube.get_tets()
        occupancies = self.net_occupancies(tets)

        mother_cube.update_by_deltas(delta_verts)
        mother_cube.update_occupancies(occupancies)

        return mother_cube


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
