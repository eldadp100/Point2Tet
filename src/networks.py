import time

import torch
import torch.nn as nn
from torch import optim
from quartet import QuarTet, Tetrahedron

import os

NEIGHBORHOOD_SIZE = 5


class MotherCubeConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(MotherCubeConv, self).__init__()
        self.lin = nn.Linear(NEIGHBORHOOD_SIZE * in_features, out_features)

    def forward(self, mother_cube: QuarTet):
        tets_vectors = []
        for tet in mother_cube:
            neighborhood_features = [tet.features]
            for n_tet in tet.neighborhood:
                neighborhood_features.append(n_tet.features)
            neighborhood_features = torch.cat(neighborhood_features, dim=0)
            tets_vectors.append(neighborhood_features)
        tets_vectors = torch.stack(tets_vectors)
        new_features = torch.relu(self.lin(tets_vectors))
        for i, tet in enumerate(mother_cube):
            tet.features = new_features[i]


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
    def __init__(self, ncf):
        super(OurNet, self).__init__()

        # ncf = [32, 64, 64, 32]  # last must be 3 because we iterate
        # self.conv_net = TetCNN_PP(ncf)  # TetCNN++
        self.net_vertices_movements = nn.Linear(ncf[0], 12)  # 3D movement
        # self.net_vertices_movements = nn.Sequential(
        #     nn.Linear(ncf[0], 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 3)
        # )
        self.net_occupancy = nn.Sequential(
            nn.Linear(ncf[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )  # Binary classifier - occupancy
        # self.net_occupancy = nn.Linear(ncf[0], 1)

    def forward(self, mother_cube):
        # 0.3-1 second
        # self.conv_net(mother_cube)

        # 2.4 seconds  ---> 0.16 seconds
        tets_features = torch.stack([tet.features for tet in mother_cube])
        tets_movements = self.net_vertices_movements(tets_features).reshape((-1, 4, 3)).cpu()
        tets_occupancy = torch.tanh(self.net_occupancy(tets_features)) / 2 + 0.5
        # tets_occupancy += 0.5 - torch.sum(tets_occupancy) / len(mother_cube)
        # tets_occupancy = torch.max(tets_occupancy, torch.tensor([0.01], device=tets_occupancy.device).expand_as(tets_occupancy))
        # tets_occupancy = torch.min(tets_occupancy, torch.tensor([0.99], device=tets_occupancy.device).expand_as(tets_occupancy))
        tets_occupancy = tets_occupancy.cpu()

        # for i, tet in enumerate(mother_cube):
            # tet.update_by_deltas(tets_movements[i])
            # tet.occupancy = tets_occupancy[i]

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
    net = OurNet(opts.ncf).to(device)
    optimizer = optim.Adam(net.parameters(), lr=opts.lr)
    scheduler = get_scheduler(opts.iterations, optimizer)

    latest_checkpoint = f'{opts.checkpoint_folder}/{opts.name}/model_checkpoint_latest.pt'
    if opts.continue_train and os.path.exists(latest_checkpoint):
        state_dict = torch.load(latest_checkpoint)
        net.load_state_dict(state_dict['net'])
        optimizer.load_state_dict(state_dict['optim'])

    return net, optimizer, scheduler


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = OurNet([3, 16, 32])
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    a = QuarTet(1, 'cpu')
    print(net(a, 0))
