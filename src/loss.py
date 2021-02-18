import torch
from chamferdist import ChamferDistance
from weighted_chamferdist import ChamferDistance as Weighted_CD


def pure_chamfer_dist(src_pc, dst_pc):
    chamferDist = ChamferDistance()
    src_pc = src_pc.type(torch.FloatTensor)
    dst_pc = dst_pc.type(torch.FloatTensor)
    dist = chamferDist.forward(src_pc, dst_pc, bidirectional=True)
    return dist


def chamfer_dist_with_weights(tetrahedrons_centers, ground_truth_point_cloud, tetrahedrons_weights):
    chamferDist = Weighted_CD()
    src_pc = tetrahedrons_centers.type(torch.FloatTensor)
    dst_pc = ground_truth_point_cloud.type(torch.FloatTensor)
    src_weights = tetrahedrons_weights.type(torch.FloatTensor)
    dist = chamferDist.forward(src_pc, dst_pc, src_weights)
    return dist


def vertices_movement_bound_loss(quartet):
    loss_1 = torch.tensor(0.)
    for v in quartet.vertices:
        sd = v.last_update_signed_distance
        loss_1 = loss_1 + torch.max(sd, torch.tensor(0.))

    return loss_1


def volumes_loss(quartet):
    volumes = [tet.volume().abs() for tet in quartet]
    avg_vols = sum(volumes) / len(volumes)
    tets_vols_loss = sum([(vol - avg_vols).abs() for vol in volumes]) / len(volumes)
    return tets_vols_loss


def vertices_movements_chamfer_loss(quartet_pts, pc):
    chamfer_loss = pure_chamfer_dist(quartet_pts.unsqueeze(0), pc.unsqueeze(0))
    return chamfer_loss


def occupancy_chamfer_loss(quartet_pts, pc):
    pass


def loss(quartet, pc):
    quartet_pts = quartet.sample_point_cloud(pc.shape[0])
    loss_monitor = {
        "vertices_movements_chamfer_loss": (0.4, vertices_movements_chamfer_loss(quartet_pts, pc)),
        "vertices_movement_bound_loss": (0.3, vertices_movement_bound_loss(quartet)),
        "volumes_loss": (0.3, volumes_loss(quartet))
    }

    print({k: f"{v[1].item() :.5f}" for k, v in loss_monitor.items()})

    return sum([lambda_i * loss_i for lambda_i, loss_i in loss_monitor.values()])

#
#     # # quartet_centers, weights = quartet.sample_point_cloud(pc.shape[0])
#     # # _, centers_weights = quartet.get_centers()
#     #
#     # # loss_1 = -chamfer_dist_with_weights(quartet_centers.unsqueeze(1), pc.unsqueeze(1), weights)
#     # # loss_2 = len(centers_weights) * (centers_weights.sum() - 1.).abs()
#     # # print(-loss_1, loss_2)
#     # # return loss_1 + loss_2
#     #
#     # quartet_pts = quartet.sample_point_cloud_3(pc.shape[0])
#     # chamfer_loss = pure_chamfer_dist(quartet_pts.unsqueeze(0), pc.unsqueeze(0))
#     #
#     # # tets volumes loss
#     #
#     # # # vertices conservativeness
#     # # vertices_loss = 0.
#     # # for v in quartet.vertices:
#     # #     vertices_loss = torch.linalg.norm(v.loc - v.original_loc) + vertices_loss
#     # # vertices_loss = vertices_loss / len(quartet.vertices)
#     # vertices_loss = quartet.last_vertex_update_average
#     # print(chamfer_loss, vertices_loss)
#     # return chamfer_loss + vertices_loss
#
#
# #
# # def chamfer_dist(src_pc, dst_pc, src_weights, padding_ratio):
# #     chamferDist = ChamferDistance()
# #     src_pc = src_pc.type(torch.FloatTensor)
# #     dst_pc = dst_pc.type(torch.FloatTensor)
# #     src_weights = src_weights.type(torch.FloatTensor)
# #     dist = chamferDist.forward(src_pc, dst_pc, src_weights, padding_ratio)
# #     return dist
#
# # def chamfer_dist(src_pc, dst_pc, src_occupancy, occ_weight=1.):
# #     chamferDist = ChamferDistance()
# #     src_pc = torch.cat([src_pc, occ_weight * src_occupancy.unsqueeze(-1)], dim=2)
# #     dst1_pc = torch.cat([dst_pc / 2., occ_weight * torch.zeros((dst_pc.shape[0], 1, 1))], dim=2)
# #     dst2_pc = torch.cat([torch.rand((dst_pc.shape[0], 1, 3)), torch.ones((dst_pc.shape[0], 1, 1))], dim=2)
# #     dst_pc = torch.cat([dst1_pc, dst2_pc], dim=0)
# #     src_pc = src_pc.type(torch.FloatTensor)
# #     dst_pc = dst_pc.type(torch.FloatTensor)
# #     dist = chamferDist.forward(src_pc, dst_pc)
# #     return dist
# #
# #
# # def pure_chamfer_dist(src_pc, dst_pc):
# #     chamferDist = ChamferDistance()
# #     src_pc = src_pc.type(torch.FloatTensor)
# #     dst_pc = dst_pc.type(torch.FloatTensor)
# #     dist = chamferDist.forward(src_pc, dst_pc)
# #     return dist
# #
# #
# # def loss(quartet, pc, n=3000, lambda_1=1., lambda_2=0., lambda_3=0.):
# #     assert lambda_1 + lambda_2 + lambda_3 <= 1.
# #     lambda_4 = 1. - lambda_1 - lambda_2 - lambda_3
# #     s = time.time()
# #     quartet_pc = quartet.sample_point_cloud_2(n)
# #     print(f"sampling time {time.time() - s}")
# #     # chamfer loss
# #     loss_1 = pure_chamfer_dist(quartet_pc.unsqueeze(1), pc.unsqueeze(1))
# #
# #     # tets volumes loss
# #     # avg_vols = sum(volumes) / len(volumes)
# #     # loss_2 = sum([(vol - avg_vols) ** 2 for vol in volumes]) / len(volumes)
# #
# #     # occupancy loss (no in between values)
# #     occupancies = [tet.occupancy for tet in quartet.curr_tetrahedrons]
# #     avg_occ = sum(occupancies) / len(occupancies)
# #     loss_3 = sum([min(occ, 1 - occ) for occ in occupancies]) / len(quartet)
# #     loss_4 = 1 / (sum([(occ - avg_occ) ** 2 for occ in occupancies]) / len(occupancies))
# #     # print(loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item())
# #     print(loss_1.item(), loss_3.item(), loss_4.item())
# #     # return lambda_1 * loss_1 + lambda_2 * loss_2 + lambda_3 * loss_3 + lambda_4 * loss_4
# #     return loss_1
#
# #
# # from pytorch3d.loss import chamfer_distance
# # def chamfer_distance_quartet_to_point_cloud(quartet, pc):
# #     quartet_pc = quartet.sample_point_cloud(pc.shape[0])
# #     out = chamfer_distance(quartet_pc.unsqueeze(1), pc.unsqueeze(1))[0]
# #     return out
#
#
# # # PUT IN COMMENT BEFORE PUSH TO GIT
# # import torch
# # def chamfer_distance_quartet_to_point_cloud(quartet, pc, quartet_N_points=3000):
# #     return quartet.sample_point_cloud(quartet_N_points).abs().sum()
# #
# # import torch
# # if __name__ == '__main__':
# #     a = torch.rand((1000, 3))
# #     b = torch.rand((1400, 3))
# #     chamfer_distance_quartet_to_point_cloud(a, b)
#
#
# def find_closest(x, pc):
#     closet_val, closet_elem = None, None
#     for y in pc:
#         if closet_val is None:
#             closet_val, closet_elem = ((x - y) ** 2).sum(), y
#         cand_val = ((x - y) ** 2).sum()
#         if cand_val < closet_val:
#             closet_val, closet_elem = cand_val, y
#     return closet_val, closet_elem
#
#
# def working_cd(pc1, pc2):
#     return torch.tensor([find_closest(x, pc1)[0] for x in pc2]).mean()
