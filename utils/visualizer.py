import open3d
import numpy as np

def visualize_quartet(quartet):
    tetra_mesh = open3d.geometry.TetraMesh()

    for tet in quartet:
        pass


    tetra_mesh = open3d.geometry.TriangleMesh.create_tetrahedron()
    open3d.visualization.draw_geometries([tetra_mesh], mesh_show_back_face=True)