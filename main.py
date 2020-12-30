import torch
from models.networks import init_net
import utils
from models.losses import chamfer_distance_quartet_to_point_cloud
from options import Options
import time
from structures.QuarTet import QuarTet

options = Options()
opts = options.args
torch.manual_seed(opts.torch_seed)
device = torch.device('cuda:{}'.format(opts.gpu) if torch.cuda.is_available() else torch.device('cpu'))
print('device: {}'.format(device))

# mesh = Mesh(opts.mesh_to_read, device=device, hold_history=True)

start_creating_quartet = time.time()
print("start creating quartet")
quartet = QuarTet(1, device)
print(f"finished creating quartet - {time.time() - start_creating_quartet} seconds")

# input point cloud
# input_xyz, input_normals = utils.read_pts(opts.input_pc)
# input_xyz = torch.Tensor(input_xyz).type(torch.FloatTensor).to(device)[None, :, :]  # .type() also changes device somewhy on the server
# input_normals = torch.Tensor(input_normals).type(torch.FloatTensor).to(device)[None, :, :]
#
# # normalize point cloud to [0,1]^3 (Unit Cube)
# input_xyz -= input_xyz.permute(1, 0).mean(dim=1)
# input_xyz /= 2 * input_xyz.permute(1, 0).max(dim=1)
# input_xyz += 0.5
# # TODO: add normals normalization

input_xyz, input_normals = torch.rand(100, 3, device=device), torch.rand(100, 3, device=device)
net, optimizer, scheduler = init_net(opts, device)

for i in range(opts.iterations):
    # TODO: Subdivide every opts.upsamp
    print(f"iteration {i} starts")
    iter_start_time = time.time()
    net(quartet, i)  # in place changes
    loss = chamfer_distance_quartet_to_point_cloud(quartet, input_xyz)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    quartet.zero_grad()
    # scheduler.step()
    print(f"iteration {i} finished - {time.time() - iter_start_time} seconds")
