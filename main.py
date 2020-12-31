import torch
from models.networks import init_net
import point2tet_utils
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
# input_xyz, input_normals = torch.rand(100, 3, device=device), torch.rand(100, 3, device=device)
input_xyz, input_normals = point2tet_utils.read_pts("pc.ply")
input_xyz = torch.Tensor(input_xyz).type(torch.FloatTensor).to(device)[None, :, :]  # .type() also changes device somewhy on the server
input_normals = torch.Tensor(input_normals).type(torch.FloatTensor).to(device)[None, :, :]
input_xyz, input_normals = input_xyz.squeeze(0), input_normals.squeeze(0)

# normalize point cloud to [0,1]^3 (Unit Cube)
input_xyz -= input_xyz.permute(1, 0).mean(dim=1)
input_xyz /= 2 * input_xyz.permute(1, 0).max(dim=1).values
input_normals /= 2 * input_xyz.permute(1, 0).max(dim=1).values
input_xyz += 0.5
input_normals += 0.5
# TODO: add normals normalization


net, optimizer, scheduler = init_net(opts, device)
evaluate_every_k_iterations = 500
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
    print(loss)
    # scheduler.step()
    print(f"iteration {i} finished - {time.time() - iter_start_time} seconds")

    if i != 0 and i % evaluate_every_k_iterations == 0:
        checkpoint_file_path = f"./checkpoints/{opts.name}/model_checkpoint_{i}.pt"
        quartet_file_path = f"./checkpoints/{opts.name}/quartet_{i}"
        mesh_file_path = f"./checkpoints/{opts.name}/mesh_{i}"

        state_dict = {
            "net": net.state_dict(),
            "optim": optimizer.state_dict()
        }

        quartet.export(quartet_file_path)
        quartet.export_mesh(mesh_file_path)
