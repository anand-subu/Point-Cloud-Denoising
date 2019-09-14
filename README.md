# Point-Cloud-Denoising

This repo is a quite basic attempt at denoising point clouds based on point wise features. 

Following features are explored in this repo:

* The use of Fast Point Feature Histograms (local features) per point as features for training using local context
  as well as obtaining voxel occupancy features  (global features) and appending whilst training to incorporate global context

* Using a simple Multi Layer Perceptron (MLP) for training the model.

Training a model to denoise point clouds is explored here by generating random noise points for each point cloud, and then
extracting features for each point cloud to train the model.

Notes:

* The noise generation employed here for generating training samples is quite simple, where a typical noise point is a tuple 
[rand(xmin,xmax),rand(ymin,ymax),rand(zmin,zmax)] where 
  * xmin,ymin,zmin- lowest coordinate values of their dimensions 
  * xmax,ymax,zmax - highest coordinate values of their dimension
 following a uniform distribution.
 
* Visualization of a noisy point cloud generated through this process yields noise points that may be spread far apart from the
original dense point cloud, making them  easily separable during the  model training due to the separability of the features calculated
for the points. Observations while training and testing prove this may be the case as model converges fast whilst yielding high accuracy, precision and recall
for the classes.

### Steps for training
```
1) Run pip install -r requirements.txt

This code requires keras(2.2.4) and tensorflow(1.10.0)
```
2) For training the model, point clouds in formats that open3d is able to parse, are supported. I've experimented with Kinect point clouds (https://rgbd-dataset.cs.washington.edu/) but any format/type is supported for this purpose.

The file train.py takes the following parameters:
  --train_with_local_features (Boolean) : If set to True, this option extracts only local features (FPFH) and trains the MLP
  --train_with_global_features (Boolean) : If set to True, this option extracts FPFH and voxel occupancy for a point cloud, and append the voxel occupancy feature vector to each FPFH feature vector. By default the voxel dimensions are 8x8x8 and the FPFH parameters are set as radius = 0.05 and max nearest neighbours = 50 inside helper.py. 
  --voxel_size : argument required only if train_with_global_features is set to True. Takes input dimensions x,y,z as comma separated string.
  --train_directory : path to directory with point clouds for training
  --test_directory : path to directory with point clouds for testing
```  
  For training with local features, run as: python train.py --train_with_local_features True \ 
                                                            --train_directory <path to train directory>\
                                                            --test_directory <path to test directory>
  
  For training with global features, run as python train.py  --train_with_global_features True \
                                                             --voxel_size 8,8,8
                                                             --train_directory <path to train directory>\
                                                             --test_directory <path to test directory>
```                                                             
Training hyperparameters can be fiddled with inside model.py and train.py to adjust batch size, epochs etc. 
Once training is finished, the model is saved directly as "final_model.h5"

### Inference
Once training is complete, inference can be performed by running as:

```
python inference.py --path_to_model <path to model h5 file> \
                    --inference_dir <path to directory with point clouds for inference>
                    --save_dir <path to save denoised point clouds>
                    --feature_type <"global" or "local">
```         
The inference code denoises point clouds and saves them to the directory.


### Visualization

The denoised point clouds are saved in standard formats, so visualization can be done using any of the standard libraries.
Open3d comes with handy packages to visualize point clouds.
