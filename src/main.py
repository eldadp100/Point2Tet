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
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def save_information(opts, i, net, optimizer, quartet):
    if i > 0:
        to_rename = f'{opts.checkpoint_folder}/{opts.name}/model_checkpoint_latest.pt'
        if os.path.exists(to_rename):
            os.rename(to_rename, f'{opts.checkpoint_folder}/{opts.name}/model_checkpoint_{i - opts.save_freq}.pt')

        checkpoint_file_path = f"{opts.checkpoint_folder}/{opts.name}/model_checkpoint_latest.pt"
        state_dict = {
            "net": net.state_dict(),
            "optim": optimizer.state_dict()
        }

        torch.save(state_dict, checkpoint_file_path)

    out_pc_file_path = f"{opts.checkpoint_folder}/{opts.name}/pc_{i}.obj"
    out_quartet_file_path = f"{opts.checkpoint_folder}/{opts.name}/quartet_{i}.tet"
    out_mesh_file_path = f"{opts.checkpoint_folder}/{opts.name}/mesh_{i}.obj"

    with open(f"{opts.checkpoint_folder}/{opts.name}/iter_num.txt", "w") as iter_num_file:
        iter_num_file.write(f"{i}")  # overide (32 bits)

    quartet.export_point_cloud(out_pc_file_path, 2500)
    quartet.export(out_quartet_file_path)
    quartet.export_mesh(out_mesh_file_path)


init_environment(opts)

start_creating_quartet = time.time()
print("start creating quartet")
quartet = QuarTet(opts.init_cube, device)
quartet.fill_sphere()
print(f"finished creating quartet - {time.time() - start_creating_quartet} seconds")

# input filled point cloud
# input_xyz, input_normals = torch.rand(100, 3, device=device), torch.rand(100, 3, device=device

pc = PointCloud()
# pc.load_file(opts.input_filled_pc)
pc.load_with_normals("../objects/g.ply")
pc.normalize()
original_input_xyz = pc.points

quartet_sdf = pc.calc_sdf(quartet.get_centers())
quartet.update_occupancy_using_sdf(quartet_sdf)

# This is how you visualize the occupied tets pc!
# pts = quartet.get_occupied_centers()
# spc = PointCloud()
# spc.init_with_points(pts)
# visualizer.visualize_pointcloud(spc)

# sample different points every iteration
chamfer_sample_size = min(original_input_xyz.shape[0], opts.chamfer_samples)
indices = np.random.randint(0, original_input_xyz.shape[0], chamfer_sample_size)
input_xyz = original_input_xyz[indices]

net, optimizer, scheduler = init_net(opts, len(quartet), device)

subdivide_spaces = 200
last_subdivide = 0
max_subdivides = 10
subdivide_up_to_now = 0

# range_init = opts.iteration_number if opts.continue_train else 1
range_init = 1
if opts.continue_train:
    if opts.iteration_number == -1:
        with open(f"{opts.checkpoint_folder}/{opts.name}/iter_num.txt") as file:
            range_init = int(file.readline().strip()) + 1
    else:
        range_init = opts.iteration_number

save_information(opts, 0, net, optimizer, quartet)
for i in range(range_init, opts.iterations + range_init + 1):

    # print(f"iteration {i} starts")
    iter_start_time = time.time()

    # # sample different points every iteration
    # chamfer_sample_size = min(original_input_xyz.shape[0], opts.chamfer_samples)
    # indices = np.random.randint(0, original_input_xyz.shape[0], chamfer_sample_size)
    # input_xyz = original_input_xyz[indices]
    # TODO: Subdivide every opts.upsamp
    net(quartet)  # in place changes
    _loss, loss_monitor = loss.loss(quartet, input_xyz)
    print({k: f"{v[1].item() :.5f}" for k, v in loss_monitor.items()})

    if i % 100 == 0:
        print({k: f"{v[1].item() :.5f}" for k, v in loss_monitor.items()})
        ######################################################
        print('occupancy gradient:')
        grad = net.net_occupancy[0].weight.grad
        if grad is not None:
            print(f'max = {grad.max()}, min = {grad.min()}')
        else:
            print('None')
        ######################################################
        print('movement gradient:')
        grad = net.net_vertices_movements[0].weight.grad
        if grad is not None:
            print(f'max = {grad.max()}, min = {grad.min()}')
        else:
            print('None')
        ######################################################
        print(f"iteration {i} finished - {time.time() - iter_start_time} seconds")

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

    # print(_loss)

    # scheduler.step()

    if i % opts.save_freq == 0:
        save_information(opts, i, net, optimizer, quartet)

    if i - last_subdivide > subdivide_spaces and loss_monitor["quartet_angles_loss"][1] == 0. and loss_monitor[
        "vertices_movement_bound_loss"][1] == 0.:
        print(" Subdivide and fix position ")

        quartet.fix_at_position()
        if subdivide_up_to_now < max_subdivides:
            quartet.subdivide_tets(net)
            print(len(quartet.curr_tetrahedrons))
            subdivide_up_to_now += 1
    else:
        quartet.reset()
    # print(f"iteration {i} finished - {time.time() - iter_start_time} seconds")

print(_loss)
