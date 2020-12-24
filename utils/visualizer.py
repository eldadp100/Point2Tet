import open3d
import numpy as np

def visualize_quartet(quartet):  
    vertex_index = 0
    vertices = []
    tetras = []

    for tet in quartet:
        tetra = []
        for vert in tet.vertices:
            vertices.append(vert.loc.cpu().detach().numpy())
            tetra.append(vertex_index)
            vertex_index += 1
        tetras.append(tetra)

    tetra_mesh = open3d.geometry.TetraMesh()
    tetra_mesh.vertices.extend(vertices)
    tetra_mesh.tetras.extend(tetras)

    open3d.visualization.draw_geometries([tetra_mesh], mesh_show_back_face=True)