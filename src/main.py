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

    target_path = os.path.join(checkpoint_folder, 'target_objects')
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    if not os.path.exists(opts.cache_folder):
        os.mkdir(opts.cache_folder)

    return checkpoint_folder, target_path


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
        iter_num_file.write(f"{i}")  # override (32 bits)

    quartet.export_point_cloud(out_pc_file_path, 2500)
    quartet.export(out_quartet_file_path)
    quartet.export_mesh(out_mesh_file_path)


checkpoint_folder, target_path = init_environment(opts)
init_cube_prefix, _ = os.path.splitext(os.path.basename(opts.init_cube))
quartet_cache_name = f'{init_cube_prefix}_quartet_data.data'
quartet_data_cache = os.path.join(opts.cache_folder, quartet_cache_name)

# if os.path.exists(quartet_data_cache):
#     print(f'found quartet data cache, loading quartet metadata from file: {quartet_data_cache}')
#     quartet = QuarTet(opts.init_cube, metadata_path=quartet_data_cache, device=device)
# else:
#     start_creating_quartet = time.time()
#     print("start creating quartet")
#     quartet = QuarTet(opts.init_cube, device)
#     print(f"finished creating quartet - {time.time() - start_creating_quartet} seconds")
#     quartet.export_metadata(quartet_data_cache)

# input filled point cloud
# input_xyz, input_normals = torch.rand(100, 3, device=device), torch.rand(100, 3, device=device

pc = PointCloud()
# pc.load_file(opts.input_filled_pc)
# pc.load_with_normals(opts.input_pc) # TODO
pc.load_file("../../filled_torus_pc.obj")
pc.normalize()
original_input_xyz = pc.points

# TODO
# quartet_sdf = pc.calc_sdf(quartet.get_centers())
# quartet.update_occupancy_using_sdf(quartet_sdf)
# quartet.fill_sphere()
quartet = QuarTet("../../torus_quartet.tet", device=device, metadata_path='default')

print("Creating target objects:")
quartet.export(path=os.path.join(target_path, 'target_quartet.tet'))
quartet.export_mesh(path=os.path.join(target_path, 'target_mesh.obj'))
pc.write_to_file(os.path.join(target_path, 'target_pc.obj'))
print("Finished")

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

subdivide_spaces = opts.upsamp
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
    chamfer_sample_size = min(original_input_xyz.shape[0], opts.chamfer_samples)
    indices = np.random.randint(0, original_input_xyz.shape[0], chamfer_sample_size)
    input_xyz = original_input_xyz[indices]
    # TODO: Subdivide every opts.upsamp
    net(quartet)  # in place changes
    _loss, loss_monitor = loss.loss(quartet, pc, input_xyz)
    print({k: f"{v[1].item() :.5f}" for k, v in loss_monitor.items()})

    # TODO
    # if i % 100 == 0:
    #     print({k: f"{v[1].item() :.5f}" for k, v in loss_monitor.items()})
    #     ######################################################
    #     print('occupancy gradient:')
    #     grad = net.net_occupancy[0].weight.grad
    #     if grad is not None:
    #         print(f'max = {grad.max()}, min = {grad.min()}')
    #     else:
    #         print('None')
    #     ######################################################
    #     print('movement gradient:')
    #     grad = net.net_vertices_movements[0].weight.grad
    #     if grad is not None:
    #         print(f'max = {grad.max()}, min = {grad.min()}')
    #     else:
    #         print('None')
    #     ######################################################
    #     print(f"iteration {i} finished - {time.time() - iter_start_time} seconds")

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

    if i % opts.save_freq == 0 and i > 0:
        save_information(opts, i, net, optimizer, quartet)

    # if i - last_subdivide > subdivide_spaces \
    #         and loss_monitor["quartet_angles_loss"][1] * loss_monitor["quartet_angles_loss"][0] == 0. \
    #         and loss_monitor["vertices_movement_bound_loss"][1] == 0.:
    #     print(" Subdivide and fix position ")
    #
    #     quartet.fix_at_position()
    #     if subdivide_up_to_now < max_subdivides:
    #         quartet.subdivide_tets(net)
    #         print(len(quartet.curr_tetrahedrons))
    #         subdivide_up_to_now += 1
    # else:
    #     quartet.reset()
    # # print(f"iteration {i} finished - {time.time() - iter_start_time} seconds")
    # if i - last_subdivide > subdivide_spaces and loss_monitor["simple_vertices_bound_loss"][1] == 0.:
    #     print(" Subdivide and fix position ")
    #
    #     quartet.fix_at_position()
    #     if subdivide_up_to_now < max_subdivides:
    #         quartet.subdivide_tets(net)
    #         print(len(quartet.curr_tetrahedrons))
    #         subdivide_up_to_now += 1
    # else:
    #     quartet.reset()
    quartet.reset()
    # print(f"iteration {i} finished - {time.time() - iter_start_time} seconds")

print(_loss)
