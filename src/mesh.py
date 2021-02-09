from queue import Queue
import copy
from pathlib import Path
import pickle
import torch
import numpy as np
import os
import uuid
import glob
from _utils import load_obj


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
