# import shutil
# import torch
# from networks import init_net
# import loss
# from options import Options
# import time
# from quartet import QuarTet
# from pointcloud import PointCloud
# import numpy as np
# import os
#
# options = Options()
# opts = options.args
# torch.manual_seed(opts.torch_seed)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # device = torch.device('cpu')
# print('device: {}'.format(device))
#
#
# def init_environment(opts):
#     if not os.path.exists(opts.checkpoint_folder):
#         os.mkdir(opts.checkpoint_folder)
#
#     checkpoint_folder = f"{opts.checkpoint_folder}/{opts.name}"
#     if os.path.exists(checkpoint_folder):
#         shutil.rmtree(checkpoint_folder)
#     time.sleep(0.1)
#     os.mkdir(checkpoint_folder)
#
#     return checkpoint_folder
#
#
# init_environment(opts)
#
# start_creating_quartet = time.time()
# print("start creating quartet")
# quartet = QuarTet(opts.init_cube, device)
# print(f"finished creating quartet - {time.time() - start_creating_quartet} seconds")
#
# # input filled point cloud
# # input_xyz, input_normals = torch.rand(100, 3, device=device), torch.rand(100, 3, device=device)
# pc = PointCloud()
# pc.load_file(opts.input_filled_pc)
# pc.normalize()
# original_input_xyz = pc.points
#
# net, optimizer, scheduler = init_net(opts, device)
#
# print(f'opts.continue_train = {opts.continue_train}')
# print(f'opts.save_freq = {opts.save_freq}')
#
#
# def save_run(i):
#     global net, optimizer, scheduler, opts
#
#     if os.path.isfile(f'{opts.checkpoint_folder}/{opts.name}/model_checkpoint_latest.pt'):
#         os.rename(f'{opts.checkpoint_folder}/{opts.name}/model_checkpoint_latest.pt',
#                   f'{opts.checkpoint_folder}/{opts.name}/model_checkpoint_{i - opts.save_freq}.pt')
#
#     checkpoint_file_path = f"{opts.checkpoint_folder}/{opts.name}/model_checkpoint_latest.pt"
#     out_pc_file_path = f"{opts.checkpoint_folder}/{opts.name}/point_clouds/pc_{i}.obj"
#     out_quartet_file_path = f"{opts.checkpoint_folder}/{opts.name}/quartets/quartet_{i}.obj"
#     out_mesh_file_path = f"{opts.checkpoint_folder}/{opts.name}/meshes/mesh_{i}.obj"
#
#     state_dict = {
#         "net": net.state_dict(),
#         "optim": optimizer.state_dict()
#     }
#
#     print('saving model')
#     torch.save(state_dict, checkpoint_file_path)
#
#     print('exporting point cloud')
#     try:
#         quartet.export_point_cloud(out_pc_file_path, 25000)
#     except IOError:
#         pass
#
#     print('saving quartet')
#     try:
#         quartet.export(out_quartet_file_path)
#     except IOError:
#         pass
#
#     print('exporting mesh')
#     try:
#         quartet.export_mesh(out_mesh_file_path)
#     except IOError:
#         pass
#
#
# for i in range(opts.iterations):
#     print(f"iteration {i} starts")
#     iter_start_time = time.time()
#
#     # sample different points every iteration
#     chamfer_sample_size = min(original_input_xyz.shape[0], opts.chamfer_samples)
#     indices = np.random.randint(0, original_input_xyz.shape[0], chamfer_sample_size)
#     input_xyz = original_input_xyz[indices]
#
#     # TODO: Subdivide every opts.upsamp
#     net(quartet)  # in place changes
#     _loss = loss.loss(quartet, input_xyz, n=chamfer_sample_size)
#     optimizer.zero_grad()
#     _loss.backward()
#     optimizer.step()
#     print(_loss)
#     # scheduler.step()
#     if i != 0 and i % opts.save_freq == 0:
#         save_run(i)
#
#     quartet.zero_grad()
#     quartet.reset()
#     print(f"iteration {i} finished - {time.time() - iter_start_time} seconds")
#
# save_run(opts.iterations)
# print(_loss)

import shutil
import torch
from networks import init_net
import loss
from options import Options
import time
from quartet import QuarTet
from pointcloud import PointCloud
import numpy as np
import os

import matplotlib.pyplot as plt
import visualizer

options = Options()
opts = options.args
torch.manual_seed(opts.torch_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))


def init_environment(opts):
    if not os.path.exists(opts.checkpoint_folder):
        os.mkdir(opts.checkpoint_folder)

    checkpoint_folder = f"{opts.checkpoint_folder}/{opts.name}"
    if os.path.exists(checkpoint_folder) and not opts.continue_train:
        shutil.rmtree(checkpoint_folder)
    time.sleep(0.1)

    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    return checkpoint_folder


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


init_environment(opts)

start_creating_quartet = time.time()

print("start creating quartet")
quartet = QuarTet(opts.init_cube, device)
print(f"finished creating quartet - {time.time() - start_creating_quartet} seconds")

# input filled point cloud
# input_xyz, input_normals = torch.rand(100, 3, device=device), torch.rand(100, 3, device=device

pc = PointCloud()
# pc.load_file(opts.input_filled_pc)
pc.load_with_normals("../objects/tiki.ply")
pc.normalize()
original_input_xyz = pc.points

quartet_sdf = pc.calc_sdf(quartet.get_centers())
quartet.update_occupancy_using_sdf(quartet_sdf)

# sample different points every iteration
chamfer_sample_size = min(original_input_xyz.shape[0], opts.chamfer_samples)
indices = np.random.randint(0, original_input_xyz.shape[0], chamfer_sample_size)
input_xyz = original_input_xyz[indices]

net, optimizer, scheduler = init_net(opts, device)

# range_init = opts.iteration_number if opts.continue_train else 1
range_init = 1
if opts.continue_train:
    if opts.iteration_number == -1:
        with open(f"{opts.checkpoint_folder}/{opts.name}/iter_num.txt") as file:
            range_init = int(file.readline().strip()) + 1
    else:
        range_init = opts.iteration_number

for i in range(range_init, opts.iterations + 1):
    print(f"iteration {i} starts")
    iter_start_time = time.time()

    # TODO: Subdivide every opts.upsamp
    net(quartet)  # in place changes
    s = time.time()
    _loss = loss.loss(quartet, input_xyz, n=chamfer_sample_size)

    ######################################################
    print('occupancy gradient:')
    grad = net.net_occupancy[0].weight.grad
    if grad is not None:
        print(f'max = {grad.max()}, min = {grad.min()}')
    else:
        print('None')
    ######################################################
    print('movement gradient:')
    grad = net.net_vertices_movements.weight.grad
    if grad is not None:
        print(f'max = {grad.max()}, min = {grad.min()}')
    else:
        print('None')
    ######################################################

    print(time.time() - s)
    optimizer.zero_grad()

    ########################
    # get_dot = grad_vis.register_hooks(_loss)
    ########################

    _loss.backward()

    # plot_grad_flow(net.named_parameters())

    ########################
    # dot = get_dot()
    # dot.save('tmp.dot')
    ########################

    # net.net_occupancy._modules['0'].weight.grad
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
        out_quartet_file_path = f"{opts.checkpoint_folder}/{opts.name}/quartet_{i}.tet"
        out_mesh_file_path = f"{opts.checkpoint_folder}/{opts.name}/mesh_{i}.obj"

        if os.path.exists(f"{opts.checkpoint_folder}/{opts.name}/iter_num.txt"):
            with open(f"{opts.checkpoint_folder}/{opts.name}/iter_num.txt") as iter_num_file:
                iter_num_file.write(i)

        state_dict = {
            "net": net.state_dict(),
            "optim": optimizer.state_dict()
        }

        torch.save(state_dict, checkpoint_file_path)
        try:
            quartet.export_point_cloud(out_pc_file_path)
        except:
            print("Error while trying to export point cloud")

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



