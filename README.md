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

# Steps for training
```
1) Run pip install -r requirements.txt
