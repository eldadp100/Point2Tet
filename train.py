import shutil
import torch
from src.networks import init_net
from src.loss import loss
from src.options import Options
import time
from src.quartet import QuarTet
from src.pointcloud import PointCloud
import numpy as np
import os

def init_environment(opts):
    torch.manual_seed(opts.torch_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    if not os.path.exists(opts.checkpoint_folder):
        os.mkdir(opts.checkpoint_folder)

    checkpoint_folder = f"{opts.checkpoint_folder}/{opts.name}"
    if os.path.exists(checkpoint_folder):
        shutil.rmtree(checkpoint_folder)
    time.sleep(0.1)
    os.mkdir(checkpoint_folder)

    return device, checkpoint_folder

def train(opts):
    device, checkpoint_folder = init_environment(opts)

    start_creating_quartet = time.time()
    print("start creating quartet")
    quartet = QuarTet(opts.init_cube, device)
    print(f"finished creating quartet - {time.time() - start_creating_quartet} seconds")

    # input filled point cloud
    # input_xyz, input_normals = torch.rand(100, 3, device=device), torch.rand(100, 3, device=device)
    pc = PointCloud()
    pc.load_file(opts.input_filled_pc)
    pc.normalize()
    original_input_xyz = pc.points

    net, optimizer, scheduler = init_net(opts, device)
    for i in range(opts.iterations):
        print(f"iteration {i} starts")
        iter_start_time = time.time()

        # sample different points every iteration
        chamfer_sample_size = min(original_input_xyz.shape[0], opts.chamfer_samples)
        indices = np.random.randint(0, original_input_xyz.shape[0], chamfer_sample_size)
        input_xyz = original_input_xyz[indices]

        # TODO: Subdivide every opts.upsamp
        net(quartet)  # in place changes
        s = time.time()
        _loss = loss(quartet, input_xyz, n=chamfer_sample_size)
        print(time.time() - s)
        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()
        quartet.zero_grad()
        print(_loss)
        # scheduler.step()

    if i != 0 and i % opts.save_freq == 0:
        to_rename = f'{opts.checkpoint_folder}/{opts.name}/model_checkpoint_latest.pt'
        if os.path.exists(to_rename):
            os.rename(to_rename, f'{opts.checkpoint_folder}/{opts.name}/model_checkpoint_{i - opts.save_freq}.pt')

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

        print(f"iteration {i} finished - {time.time() - iter_start_time} seconds")

    print(_loss)
    return _loss.item()

if __name__ == "__main__":
    options = Options()
    opts = options.args
    train(opts)