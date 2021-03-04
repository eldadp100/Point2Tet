import numpy as np
import torch
import uuid
import os


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


def random_file_name(ext, prefix='temp'):
    return f'{prefix}{uuid.uuid4()}.{ext}'


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


def create_torus_point_cloud(major_radius=0.5, minor_radius=None, aspect_ratio=3., offset=(0.5, 0.5, 0.), filename='torus.obj',
                 N=int(1e4), normals=True):
    """
    Creates a torus point cloud (only shell), and writes it to an .obj or .ply file
    :param major_radius: the major radius of the torus (the distance between center of donut's perimeter and the center of the donut)
    :param minor_radius: the minor radius of the torus (the radius of the donut's perimeter). if None determined by the major radius and the aspect ratio
    :param aspect_ratio: the ratio between major and minor radiuses (major / minor). if the minor radius is given this parameter is ignored
    :param offset: a tuple of size 3 containing offset to the torus on the x, y, z axis
    :param filename: the file name to export the created torus to; if None no file is created
    :param N: the number of points to sample
    :param normals: if True exports the point cloud with normals
    :return: a torch.tensor containing the points (of shape [N, 3])
    """
    if minor_radius is None:
        minor_radius = major_radius / aspect_ratio

    rng = np.random.default_rng()
    thetas, phis = rng.uniform(high=360., size=N), rng.uniform(high=360., size=N)

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
    if normals:
        normal_vectors = torch.stack([get_normal(ct, cp, st, sp) for ct, cp, st, sp in zip(cts, cps, sts, sps)])


    if filename is not None:
        split = os.path.splitext(filename)
        if len(split) > 1:
            if split[1] == '.obj':
                export_obj(result, normal_vectors, filename)
            elif split[1] == '.ply':
                export_ply(result, normal_vectors, filename)
            else:
                raise IOError("Error: File format not supported!")

    return result



if __name__ == '__main__':
    # vs, faces = load_obj('../objects/cube.obj', normalize=True)
    # export('../objects/normalized_cube.obj', vs, faces)
    create_torus_point_cloud(filename="torus_pc.ply", normals=True)