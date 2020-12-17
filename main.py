import torch
from models.networks import init_net
import utils
from models.losses import chamfer_distance
from options import Options
import time
import os

from structures.QuarTet import QuarTet

options = Options()
opts = options.args

torch.manual_seed(opts.torch_seed)
device = torch.device('cuda:{}'.format(opts.gpu) if torch.cuda.is_available() else torch.device('cpu'))
print('device: {}'.format(device))

# mesh = Mesh(opts.mesh_to_read, device=device, hold_history=True)

quartet = QuarTet(10)

# input point cloud
input_xyz, input_normals = utils.read_pts(opts.input_pc)
input_xyz = torch.Tensor(input_xyz).type(torch.FloatTensor).to(device)[None, :, :]
input_normals = torch.Tensor(input_normals).type(torch.FloatTensor).to(device)[None, :, :]

# normalize point cloud to [0,1]^3 (Unit Cube)
input_xyz -= input_xyz.permute(1, 0).mean(dim=1)  # TODO: Here might be a bug
input_xyz /= 2 * input_xyz.permute(1, 0).max(dim=1)  # TODO: Here might be a bug
input_xyz += 0.5

net, optimizer, scheduler = init_net(opts, device, )

for i in range(opts.iterations):
    # TODO: Subdivide every opts.upsamp

    iter_start_time = time.time()
    # Apply network: quartet = net(quartet)
    # loss = calculate chamfer
    # update network

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()

    iter_end_time = time.time()

    # print iteration summary
    # if i % 1 == 0:
    #     print(f'{os.path.basename(opts.input_pc)}; iter: {i} out of: {opts.iterations}; loss: {loss.item():.4f};'
    #           f' sample count: {num_samples}; time: {end_time - start_time:.2f}')
    # if i % opts.export_interval == 0 and i > 0:
    #     print('exporting reconstruction... current LR: {}'.format(optimizer.param_groups[0]['lr']))
    #     with torch.no_grad():
    #         part_mesh.export(os.path.join(opts.save_path, f'recon_iter_{i}.obj'))
    #
    # if (i > 0 and (i + 1) % opts.upsamp == 0):
    #     mesh = part_mesh.main_mesh
    #     num_faces = int(np.clip(len(mesh.faces) * 1.5, len(mesh.faces), opts.max_faces))
    #
    #     if num_faces > len(mesh.faces) or opts.manifold_always:
    #         # up-sample mesh
    #         mesh = utils.manifold_upsample(mesh, opts.save_path, Mesh,
    #                                        num_faces=min(num_faces, opts.max_faces),
    #                                        res=opts.manifold_res, simplify=True)
    #
    #         part_mesh = PartMesh(mesh, num_parts=options.get_num_parts(len(mesh.faces)), bfs_depth=opts.overlap)
    #         print(f'upsampled to {len(mesh.faces)} faces; number of parts {part_mesh.n_submeshes}')
    #         net, optimizer, rand_verts, scheduler = init_net(mesh, part_mesh, device, opts)
    #         if i < opts.beamgap_iterations:
    #             print('beamgap updated')
    #             beamgap_loss.update_pm(part_mesh, input_xyz)

with torch.no_grad():
    quartet.export(os.path.join(opts.save_path, 'last_recon.obj'))
