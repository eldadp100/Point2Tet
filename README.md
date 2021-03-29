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

# Important CLI Parameters
1. --name: The name of the generated folder in which the data will be saved
2. --save_freq: The frequency to save the model and export the resulting point cloud, mesh and tetrahedral mesh object
3. --input_cube: Path to the file containing the file to load as the initial tetrahedral mesh (a cube is assumed in .tet format)
4. --input_filled_pc: Path to the filled point cloud to use as input

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
