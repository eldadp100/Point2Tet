# Point2Tet
# Abstract
The problem of transformation from point cloud to other representation techniques (i.e. mesh or signed distance functions) is a well known problem in computer graphics.
In this work we show how to translate a point cloud into a corresponds tetrahedral mesh.
We also deal with the movement crossing problem, define the TG data structure and define a convolutional network which utilizes the geometrical attributes of tetrahedral meshes.

# Tetrahedral Mesh
The Tetrahedral Mesh is a structure made of tetrahedrons occupying the entire space of the shape. It has a lot of practical applications, mainly in physics simulations, 
as it can represent volumetric shapes with more accuracy than voxels (as they are fixed in shape), and compared to Point Clouds or regular triangular Meshes, that posses no intrensic volumetric properties.
![alt text](http://url/to/img.png)
The way we use this structure to learn an object is by setting an occupancy value for each tetrahedron (determening whether the tetrhedron is inside or outside the shape - occupancy value of 1 is inside and 0 is outside) and then learning to move each vertex so the shell created by the tetrahedron best fits the input point cloud.
The shell created by the structure described above is the boundry layer between the occupied (occupancy value of 1) and unoccupied (occupancy value of 0) tetrahedrons. Note that becuase the tetrahdrons are made out of triangles, this shell is actually a regular triangular mesh. 
This property allowes us to use the same arcitecture to convert Point Clouds to Meshes.

# Input Point Cloud
In this work we assume the input point cloud is filled, that is there are points sampled from within the object and not only from the surface. The reason we assume that is so we can initialize the occupancy values (as described above) to suit the inside and outside of the shape. Note that in the paper we describe ways to fill a point cloud, therefore rendering this project even more versatile.
We also describe a possible way to implement a different network to learn the occupancies from a given point cloud.

# Network Architecture
As part of this work, we designed and implemented a convolutional network that operates on a tetrahedral mesh. In order to learn features of the tetrahedrons, we save for each tetrahedron a feature vector which is updated by the network and used later to learn vertices displacement (and optionally the occupancy values).
The network consists of an embedding layer, a convolutional layer and a linear layer. We also implemented a pooling layer, allowing for the reduction of the number of tetrahedrons, but we didn't yet implement an unpooling layer and therefore decided not to include that as part of the final network (but we did include this as part of the code).

## Tetrahedral Convolution
The most novel aspect of this work is the tetrahedral convolution layer, which allows us to implement the tried and tested method of convoluting over the input and aggregating the learned features local features (meaning each feature vector can be influenced by the local area, and not only itself).
The idea behind this convolution, and the property that makes it possible, is the fact that each tetrahedron has 4 neighbors (except the boundry tetrahedrons on the outer layer of the tetrahedral mesh object; a solution to that is described in the paper, but the basic idea is to use the features of the same tetrahedron to 'complete' the neighborhood to size 4).
Therefore in order to implement the convolution, all we had to do is take the neighborhood of the tetrahedron (its 4 neighbors) and aggregate their features (in our case we used mean).
