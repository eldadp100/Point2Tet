import time
import torch
from chamferdist import ChamferDistance


def chamfer_dist(src_pc, dst_pc, src_weights):
    chamferDist = ChamferDistance()
    src_pc = src_pc.type(torch.FloatTensor)
    dst_pc = dst_pc.type(torch.FloatTensor)
    src_weights = src_weights.type(torch.FloatTensor)
    dist = chamferDist.forward(src_pc, dst_pc, src_weights)
    return dist


def loss(quartet, pc, n=3000, lambda_1=1., lambda_2=0., lambda_3=0.):
    assert lambda_1 + lambda_2 + lambda_3 <= 1.
    lambda_4 = 1. - lambda_1 - lambda_2 - lambda_3
    quartet_pc, weights = quartet.sample_point_cloud(n)

    # chamfer loss
    loss_1 = chamfer_dist(quartet_pc.unsqueeze(1), pc.unsqueeze(1), weights)

    # tets volumes loss
    # avg_vols = sum(volumes) / len(volumes)
    # loss_2 = sum([(vol - avg_vols) ** 2 for vol in volumes]) / len(volumes)

    # occupancy loss (no in between values)
    occupancies = [tet.occupancy for tet in quartet.curr_tetrahedrons]
    avg_occ = sum(occupancies) / len(occupancies)
    loss_3 = sum([min(occ, 1 - occ) for occ in occupancies]) / len(quartet)
    loss_4 = 1 / sum([(occ - avg_occ) ** 2 for occ in occupancies]) / len(occupancies)
    # print(loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item())
    print(loss_1.item(), loss_3.item(), loss_4.item())
    # return lambda_1 * loss_1 + lambda_2 * loss_2 + lambda_3 * loss_3 + lambda_4 * loss_4
    return 100 * loss_1 + loss_4 + loss_3

#
# from pytorch3d.loss import chamfer_distance
# def chamfer_distance_quartet_to_point_cloud(quartet, pc):
#     quartet_pc = quartet.sample_point_cloud(pc.shape[0])
#     out = chamfer_distance(quartet_pc.unsqueeze(1), pc.unsqueeze(1))[0]
#     return out


# # PUT IN COMMENT BEFORE PUSH TO GIT
# import torch
# def chamfer_distance_quartet_to_point_cloud(quartet, pc, quartet_N_points=3000):
#     return quartet.sample_point_cloud(quartet_N_points).abs().sum()
#
# import torch
# if __name__ == '__main__':
#     a = torch.rand((1000, 3))
#     b = torch.rand((1400, 3))
#     chamfer_distance_quartet_to_point_cloud(a, b)
