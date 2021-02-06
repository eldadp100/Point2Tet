import open3d

def visualize_quartet(quartet):
    mesh = open3d.geometry.TetraMesh()
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

    o3d_tetra_mesh = open3d.geometry.TetraMesh()
    o3d_tetra_mesh.vertices.extend(vertices)
    o3d_tetra_mesh.tetras.extend(tetras)
    mesh += o3d_tetra_mesh

    open3d.visualization.draw_geometries([mesh])

def visualize_pointcloud(pc):
    points = pc.points.cpu().detach().numpy()
    points = open3d.utility.Vector3dVector(points)
    o3d_pc = open3d.geometry.PointCloud(points)
    
    open3d.visualization.draw_geometries([o3d_pc])
