import shutil
import torch
from networks import init_net
from loss import chamfer_distance_quartet_to_point_cloud
from options import Options
import time
from quartet import QuarTet
from pointcloud import PointCloud
import numpy as np
import os

options = Options()
opts = options.args
torch.manual_seed(opts.torch_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))


def init_environment(opts):
    if not os.path.exists(opts.checkpoint_folder):
        os.mkdir(opts.checkpoint_folder)

    checkpoint_folder = f"{opts.checkpoint_folder}/{opts.name}"
    if os.path.exists(checkpoint_folder):
        shutil.rmtree(checkpoint_folder)
    time.sleep(0.1)
    os.mkdir(checkpoint_folder)

    return checkpoint_folder


init_environment(opts)

start_creating_quartet = time.time()
print("start creating quartet")
quartet = QuarTet(opts.init_cube, device)
print(f"finished creating quartet - {time.time() - start_creating_quartet} seconds")

# input filled point cloud
# input_xyz, input_normals = torch.rand(100, 3, device=device), torch.rand(100, 3, device=device)
pc = PointCloud()
pc.load_file(opts.input_filled_pc)
pc.normalize()
input_xyz = pc.points

N = 20000
indices = np.random.randint(0, input_xyz.shape[0], N)
input_xyz = input_xyz[indices]

net, optimizer, scheduler = init_net(opts, device)
for i in range(opts.iterations):
    # TODO: Subdivide every opts.upsamp
    print(f"iteration {i} starts")
    iter_start_time = time.time()
    net(quartet, 0)  # in place changes
    _loss = chamfer_distance_quartet_to_point_cloud(quartet, input_xyz, quartet_N_points=N)
    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()
    quartet.zero_grad()
    print(_loss)
    # scheduler.step()
    print(f"iteration {i} finished - {time.time() - iter_start_time} seconds")

    if i != 0 and i % opts.save_freq == 0:
        os.rename(f'{opts.checkpoint_folder}/{opts.name}/model_checkpoint_latest.pt',
                  f'{opts.checkpoint_folder}/{opts.name}/model_checkpoint_{i - opts.save_freq}.pt')

        checkpoint_file_path = f"{opts.checkpoint_folder}/{opts.name}/model_checkpoint_latest.pt"
        out_pc_file_path = f"{opts.checkpoint_folder}/{opts.name}/pc_{i}.obj"
        out_quartet_file_path = f"{opts.checkpoint_folder}/{opts.name}/quartet_{i}.obj"
        out_mesh_file_path = f"{opts.checkpoint_folder}/{opts.name}/mesh_{i}.obj"

        state_dict = {
            "net": net.state_dict(),
            "optim": optimizer.state_dict()
        }

        torch.save(state_dict, checkpoint_file_path)
        try:
            quartet.export_point_cloud(out_pc_file_path, 25000)
        except:
            pass

        try:
            quartet.export(out_quartet_file_path)
        except:
            pass

        try:
            quartet.export_mesh(out_mesh_file_path)
        except:
            pass

    quartet.reset()
