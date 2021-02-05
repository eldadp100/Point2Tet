from queue import Queue
import copy
from pathlib import Path
import pickle
import torch
import numpy as np
import os
import uuid
import glob


def load_obj(file):
    vs, faces = [], []
    f = open(file)
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
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


class Mesh:

    def __init__(self, file):
        self.vs, self.faces = load_obj(file)
        faces_list = []
        for f in self.faces:
            v1, v2, v3 = self.vs[f[0]], self.vs[f[1]], self.vs[f[2]]
            if sum(v1 == v2) == 3 or sum(v1 == v3) == 3 or sum(v2 == v3) == 3:
                continue
            faces_list.append((v1, v2, v3))
        self.faces = np.array(faces_list)
