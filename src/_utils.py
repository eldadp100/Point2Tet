import os
import shutil
import time

import numpy as np
import torch


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


def create_torus_point_cloud(major_radius=0.5, minor_radius=None, aspect_ratio=3., offset=(0.5, 0.5, 0.),
                             filename='torus.obj',
                             N=int(1e4), normals=True, filled=False):
    """
    Creates a torus point cloud (only shell), and writes it to an .obj or .ply file
    :param major_radius: the major radius of the torus (the distance between center of donut's perimeter and the center of the donut)
    :param minor_radius: the minor radius of the torus (the radius of the donut's perimeter). if None determined by the major radius and the aspect ratio
    :param aspect_ratio: the ratio between major and minor radiuses (major / minor). if the minor radius is given this parameter is ignored
    :param offset: a tuple of size 3 containing offset to the torus on the x, y, z axis
    :param filename: the file name to export the created torus to; if None no file is created
    :param N: the number of points to sample
    :param normals: if True exports the point cloud with normals
    :param filled: if True creates a filled torus; if both normals and filled are True creates a filled torus without normals
    :return: a torch.tensor containing the points (of shape [N, 3])
    """
    result, normals = None, None

    if minor_radius is None:
        minor_radius = major_radius / aspect_ratio

    if filled:
        minor_radiuses = [(a / 10) * minor_radius for a in range(1, 10)]
        points_count = [int((a ** 2 / 100) * N) for a in range(1, 10)]
        normals = False

    def create_torus(minor_radius, points_count):
        rng = np.random.default_rng()
        thetas, phis = rng.uniform(high=360., size=points_count), rng.uniform(high=360., size=points_count)

        def get_point(ct, cp, st, sp):
            """ ct = cos(theta); cp = cos(phi); st = sin(theta); sp = sin(phi) """
            temp = major_radius + minor_radius * ct
            return torch.cat([
                torch.tensor([temp * cp]),
                torch.tensor([temp * sp]),
                torch.tensor([minor_radius * st])
            ])

        def get_normal(ct, cp, st, sp):
            temp = minor_radius * ct
            return torch.cat([
                torch.tensor([temp * cp]),
                torch.tensor([temp * sp]),
                torch.tensor([minor_radius * st])
            ])

        cts, cps, sts, sps = np.cos(thetas), np.cos(phis), np.sin(thetas), np.sin(phis)
        result = torch.stack([get_point(ct, cp, st, sp) for ct, cp, st, sp in
                              zip(cts, cps, sts, sps)])
        result += torch.tensor(offset)
        normal_vectors = None
        if normals:
            normal_vectors = torch.stack([get_normal(ct, cp, st, sp) for ct, cp, st, sp in zip(cts, cps, sts, sps)])
        return result, normal_vectors

    result, normals = create_torus(minor_radius, N)
    if filled:
        for r, n in zip(minor_radiuses, points_count):
            new_result, new_normals = create_torus(r, n)
            result = torch.cat([result, new_result])
            if normals:
                normals = torch.cat([normals, new_normals])

    if filename is not None:
        split = os.path.splitext(filename)
        if len(split) > 1:
            if split[1] == '.obj':
                export_obj(result, normals, filename)
            elif split[1] == '.ply':
                export_ply(result, normals, filename)
            else:
                raise IOError("Error: File format not supported!")

    return result


def create_sphere_filled_point_cloud():
    N = 10000
    sphere_points = []
    while len(sphere_points) < N:
        rand_point = np.random.rand(3)
        if np.linalg.norm(rand_point - 0.5) <= 0.5:
            sphere_points.append(rand_point)

    # out = PointCloud()
    # out.init_with_points(torch.from_numpy(np.array(sphere_points)))
    # out.write_to_file('./filled_sphere.obj')
    export_obj(torch.from_numpy(np.array(sphere_points)), None, './filled_sphere.obj')


def read_pts(pts_file):
    """
    :param pts_file: file path of a plain text list of points
    such that a particular line has 6 float values: x, y, z, nx, ny, nz
    which is typical for (plaintext) .ply or .xyz
    :return: xyz, normals
    """
    xyz, normals = [], []
    with open(pts_file, 'r') as f:
        # line = f.readline()
        # spt = f.read().split('\n')
        # while line:
        # for line in spt:
        for line in f:
            parts = line.strip().split(' ')
            try:
                x = np.array(parts, dtype=np.float32)
                xyz.append(x[:3])
                normals.append(x[3:])
            except:
                pass
    return np.array(xyz, dtype=np.float32), np.array(normals, dtype=np.float32)


def load_obj(file, normalize=False):
    vs, faces = [], []
    with open(file) as f:
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:4]])
            elif splitted_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                                   for ind in face_vertex_ids]
                faces.append(face_vertex_ids)
    vs = np.asarray(vs)
    if normalize:
        vs -= vs.mean(axis=0)
        vs += np.abs(vs.min(axis=0))
        vs /= vs.max(axis=0)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


def export(file, vs, faces, vn=None, color=None):
    with open(file, 'w+') as f:
        for vi, v in enumerate(vs):
            if color is None:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            else:
                f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
            if vn is not None:
                f.write("vn %f %f %f\n" % (vn[vi, 0], vn[vi, 1], vn[vi, 2]))
        for face in faces:
            f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))


def export_obj(vertices, normals, filename):
    with open(filename, 'w') as f:
        if normals is not None:
            for point, normal in zip(vertices, normals):
                x, y, z = point
                f.write(f"v {x.item()} {y.item()} {z.item()}\n")
                xn, yn, zn = normal
                f.write(f"vn {xn.item()} {yn.item()} {zn.item()}\n")
        else:
            for point in vertices:
                x, y, z = point
                f.write(f"v {x.item()} {y.item()} {z.item()}\n")


def export_ply(vertices, normals, filename):
    with open(filename, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}")
        if normals is not None:
            f.write('\nproperty double '.join(['', 'x', 'y', 'z', 'nx', 'ny', 'nz']))
            # f.write('nz\n')
        else:
            f.write('\nproperty double '.join(['', 'x', 'y', 'z']))
        f.write('\nend_header\n')
        if normals is not None:
            for point, normal in zip(vertices, normals):
                x, y, z = point
                xn, yn, zn = normal
                f.write(f"{x.item()} {y.item()} {z.item()} {xn.item()} {yn.item()} {zn.item()}\n")
        else:
            for point in vertices:
                x, y, z = point
                f.write(f"{x.item()} {y.item()} {z.item()}\n")


if __name__ == '__main__':
    # vs, faces = load_obj('../objects/cube.obj', normalize=True)
    # export('../objects/normalized_cube.obj', vs, faces)
    # create_torus_point_cloud(filename="../objects/filled_torus_pc.ply", filled=True)
    create_torus_point_cloud(filename="../objects/torus_pc.ply", normals=True)
