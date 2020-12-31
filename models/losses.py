import torch
from chamferdist import ChamferDistance

def chamfer_dist(src_pc, dst_pc):
    chamferDist = ChamferDistance()
    src_pc = src_pc.type(torch.FloatTensor)
    dst_pc = dst_pc.type(torch.FloatTensor)
    dist_forward = chamferDist(src_pc, dst_pc)
    return dist_forward.detach().cpu().item()

def chamfer_dist_quartet_to_pc(quartet, pc, quartet_sample_count=3000):
    src_pc = quartet.sample_point_cloud(quartet_sample_count)
    dst_pc = pc.points
    return chamfer_dist(src_pc.unsqueeze(1), dst_pc.unsqueeze(1))
