# Point2Tet
Point2Tet is a novel deep neural network to learn Tetrahedral Meshed from 3d point clouds.

<table>
  <tr><th>Tetrahedral Mesh of a Cube</th></tr>
  <tr><td><img src="/images/quartet_0.1.png" /></td></tr>
</table>

# Getting Started
* Clone or download this repo.
* Install dependecies (requires PyTorch).
* Run main.py using Python (version 3.x.x).

## Important CLI Parameters
1. --name: The name of the generated folder in which the data will be saved
2. --input_cube: Controls resolution. Path to the file containing the file to load as the initial tetrahedral mesh (a cube is assumed in .tet format)
3. --input_filled_pc: Path to the filled point cloud to use as input. (.obj format)
4. --input_surface_pc: [Optional] Path to the surface point cloud (without normals) to use as additional input. (.obj format)

## Examples
1. sphere: python main.py --name sphere --init_cube ../objects/cube_0.05.tet --input_filled_pc ../objects/filled_sphere.obj --input_surface_pc ../objects/surface_sphere.obj 
2. torus: python main.py --name sphere --init_cube ../objects/cube_0.05.tet --input_filled_pc ../objects/filled_torus.obj --input_surface_pc ../objects/surface_torus.obj
3. G: python main.py --name sphere --init_cube ../objects/cube_0.05.tet --input_filled_pc ../objects/filled_g.obj --input_surface_pc ../objects/surface_g.obj
* Execute from src folder.
# Results
<table>
  <tr>
    <th>Sphere</th>
    <th>Torus</th>
    <th>G</th>
  </tr>
  <tr>
    <td width=30%><img src="/images/sphere2.JPG" width=100% /></td>
    <td width=30%><img src="/images/torus_0.05.JPG" width=100% /></td>
    <td width=30%><img src="/images/g005.JPG" width=100% /></td>
  </tr>
</table>
