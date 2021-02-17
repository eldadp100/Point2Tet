import time
import torch
from chamferdist import ChamferDistance
from wighted_chamferdist import ChamferDistance as Weighted_CD



def chamfer_dist_with_weights(tetrahedrons_centers, ground_truth_point_cloud, tetrahedrons_weights):
    chamferDist = Weighted_CD()
    src_pc = tetrahedrons_centers.type(torch.FloatTensor)
    dst_pc = ground_truth_point_cloud.type(torch.FloatTensor)
    src_weights = tetrahedrons_weights.type(torch.FloatTensor)
    dist = chamferDist.forward(src_pc, dst_pc, src_weights)
    return dist


def loss(quartet, pc):
    quartet_centers, weights = quartet.sample_point_cloud(pc.shape[0])
    _, centers_weights = quartet.get_centers()

    loss_1 = -chamfer_dist_with_weights(quartet_centers.unsqueeze(1), pc.unsqueeze(1), weights)
    loss_2 = len(centers_weights) * (centers_weights.sum() - 1.).abs()
    print(-loss_1, loss_2)
    return loss_1 + loss_2




#
# def chamfer_dist(src_pc, dst_pc, src_weights, padding_ratio):
#     chamferDist = ChamferDistance()
#     src_pc = src_pc.type(torch.FloatTensor)
#     dst_pc = dst_pc.type(torch.FloatTensor)
#     src_weights = src_weights.type(torch.FloatTensor)
#     dist = chamferDist.forward(src_pc, dst_pc, src_weights, padding_ratio)
#     return dist

# def chamfer_dist(src_pc, dst_pc, src_occupancy, occ_weight=1.):
#     chamferDist = ChamferDistance()
#     src_pc = torch.cat([src_pc, occ_weight * src_occupancy.unsqueeze(-1)], dim=2)
#     dst1_pc = torch.cat([dst_pc / 2., occ_weight * torch.zeros((dst_pc.shape[0], 1, 1))], dim=2)
#     dst2_pc = torch.cat([torch.rand((dst_pc.shape[0], 1, 3)), torch.ones((dst_pc.shape[0], 1, 1))], dim=2)
#     dst_pc = torch.cat([dst1_pc, dst2_pc], dim=0)
#     src_pc = src_pc.type(torch.FloatTensor)
#     dst_pc = dst_pc.type(torch.FloatTensor)
#     dist = chamferDist.forward(src_pc, dst_pc)
#     return dist
#
#
# def pure_chamfer_dist(src_pc, dst_pc):
#     chamferDist = ChamferDistance()
#     src_pc = src_pc.type(torch.FloatTensor)
#     dst_pc = dst_pc.type(torch.FloatTensor)
#     dist = chamferDist.forward(src_pc, dst_pc)
#     return dist
#
#
# def loss(quartet, pc, n=3000, lambda_1=1., lambda_2=0., lambda_3=0.):
#     assert lambda_1 + lambda_2 + lambda_3 <= 1.
#     lambda_4 = 1. - lambda_1 - lambda_2 - lambda_3
#     s = time.time()
#     quartet_pc = quartet.sample_point_cloud_2(n)
#     print(f"sampling time {time.time() - s}")
#     # chamfer loss
#     loss_1 = pure_chamfer_dist(quartet_pc.unsqueeze(1), pc.unsqueeze(1))
#
#     # tets volumes loss
#     # avg_vols = sum(volumes) / len(volumes)
#     # loss_2 = sum([(vol - avg_vols) ** 2 for vol in volumes]) / len(volumes)
#
#     # occupancy loss (no in between values)
#     occupancies = [tet.occupancy for tet in quartet.curr_tetrahedrons]
#     avg_occ = sum(occupancies) / len(occupancies)
#     loss_3 = sum([min(occ, 1 - occ) for occ in occupancies]) / len(quartet)
#     loss_4 = 1 / (sum([(occ - avg_occ) ** 2 for occ in occupancies]) / len(occupancies))
#     # print(loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item())
#     print(loss_1.item(), loss_3.item(), loss_4.item())
#     # return lambda_1 * loss_1 + lambda_2 * loss_2 + lambda_3 * loss_3 + lambda_4 * loss_4
#     return loss_1

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
