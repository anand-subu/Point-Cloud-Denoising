# Point-Cloud-Denoising

This was a straight-forward attempt at attempting to train a classifier to identify Gaussian noise point occurrences in 3D Kinect Point Clouds. Noise handling and outlier detection is often an important task to handle with respect to efficient scanning.

# Salient Features
3D Kinect Point clouds were considered, and Gaussian noise points, subjective to the range of each file was considered and added. Point-wise features were obtained by usage of the fast-point feature histograms, and global context was obtained as a Bag-of-Words Representation and concatenated with each point-wise descriptor. The input is fed to a Neural Network for training. For purposes of labelling, noise points are assigned the value '1' and the raw points ,'0'

Bit of an overkill to jump to neural networks as the classifier of choice in this context but the main intention was to investigate if the NN was able to perform better with addition of global features. To scale down on timing involved, SVMs/Random Forests could also be trained for the same data.

The dataset isn't included here, but can be downloaded at https://rgbd-dataset.cs.washington.edu/dataset/. Various types are available, feel free to experiment with a different combination of point clouds.

# Dependencies
1)Open3d
2)Tensorflow
3)Scikit
4)Numpy

# Structure

After cloning this git and making sure your requirments are fine, create a folder called 'training' and move your training data as .pcd files here. Run helper.py with the arguement --path which is the path to the directory holding your training files. The module performs feature extraction and saves the training data as .npy arrays.

Run training.py to start training the NN using tensorflow. The training converges pretty quickly,for the training set, but has a harder job in general with points with irregular orientations/objects with large gaps between them.



 




