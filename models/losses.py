import torch
from chamferdist import ChamferDistance

def chamfer_dist(src_pc, dst_pc):
    chamferDist = ChamferDistance()
    src_pc = src_pc.type(torch.FloatTensor)
    dst_pc = dst_pc.type(torch.FloatTensor)
    dist_forward = chamferDist(src_pc, dst_pc)
    return dist_forward.detach().cpu().item()

def chamfer_distance_quartet_to_point_cloud(quartet, pc, i, quartet_N_points=3000):
    # quartet_pc = quartet.sample_point_cloud(quartet_N_points)
    # return chamfer_distance(quartet_pc.unsqueeze(1), pc.unsqueeze(1))
    # return quartet.sample_point_cloud(quartet_N_points).abs().sum() / quartet_N_points
    # for tet in quartet:
    #     for vert in tet.vertices:
    #         print(vert.loc)
    loss = torch.stack([tet.vertices[0].loc for tet in quartet]).abs().sum() / 1000
    if i % 100 != 0:
        for tet in quartet:
            tet.update_by_deltas(-tet.last_move)
    return loss
    # return torch.stack([tet.vertices[0].loc for tet in quartet]).abs().sum() / 1000

# from pytorch3d.loss import chamfer_distance
# def chamfer_distance_quartet_to_point_cloud(quartet, pc):
#     quartet_pc = quartet.sample_point_cloud(pc.shape[0])
#     out = chamfer_distance(quartet_pc.unsqueeze(1), pc.unsqueeze(1))[0]
#     return out
# # PUT IN COMMENT BEFORE PUSH TO GIT
# import torch
#
#
# def chamfer_distance_quartet_to_point_cloud(quartet, pc, quartet_N_points=3000):
#     return quartet.sample_point_cloud(quartet_N_points).abs().sum()
#
# import torch
# if __name__ == '__main__':
#     a = torch.rand((1000, 3))
#     b = torch.rand((1400, 3))
#     chamfer_distance_quartet_to_point_cloud(a, b)
