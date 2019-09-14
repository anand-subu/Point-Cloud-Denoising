import open3d as o3d
import numpy as np
import random
import os
import pyntcloud
from tqdm import tqdm
from open3d import *
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization


def reg_noise(pts,num_of_points):
    
    """ 
    Method to add noise points to point cloud
    Generates noise points in the range of 
    the point to the max coordinate in that dimension
    """
    x,y,z = pts.min(axis=0)
    x_max,y_max,z_max = pts.max(axis=0)
    noise = []
    for i in range(num_of_points):
        x_noise = random.uniform( x, x_max )
        y_noise = random.uniform( y, y_max )
        z_noise = random.uniform( z, z_max )
        noise+=[[x_noise,y_noise,z_noise]]
    
    return np.concatenate((pts,np.asarray(noise)))


def inference_feature_extraction(point_cloud_path,feature_flag):
    
    """
    Feature extraction for inference

    """    
    if feature_flag == "local":
        
        point_cloud = read_point_cloud(point_cloud_path)
        estimate_normals(point_cloud,KDTreeSearchParamHybrid(radius=0.01,max_nn=30))
        fpfh_features=compute_fpfh_feature(point_cloud,KDTreeSearchParamHybrid(radius=0.05,max_nn=50))
        features=fpfh_features.data.T
        features=features/np.max(features)
        
        return features
        
    elif feature_flag == "global":
        features_global=[]
        point_cloud = read_point_cloud(point_cloud_path)
        estimate_normals(point_cloud,KDTreeSearchParamHybrid(radius=0.01,max_nn=30))
        fpfh_features = compute_fpfh_feature(point_cloud,KDTreeSearchParamHybrid(radius=0.05,max_nn=50))
        features = fpfh_features.data.T
        features = features/np.max(features)

        voxel_features=voxel_occupancy_features(point_cloud_path)

        for item in features:
            features_global.append(np.append(item,voxel_features,axis=0))
   
        return np.array(features_global)
    

def feature_extraction(directory,num_of_noise_points=1000):

    """
    Extracting the fast point feature histograms 
    from the point clouds
    """

    features=[]
    labels=[]
    if os.path.isdir("Noisy") is False:
        os.mkdir("Noisy")
    i=0
    
    print("Adding noise to point clouds and extracting features (Only pointwise)")
    
    for files in tqdm(os.listdir(directory)):

        pcd = read_point_cloud(os.path.join(directory,files))
        points = np.asarray(pcd.points)
        
        noise_points=reg_noise(points,num_of_noise_points)

        pcd.points=Vector3dVector(noise_points)

        write_point_cloud(os.path.join("Noisy",files)+'_noisy.pcd',pcd)

        labels_temp=np.zeros((1,len(points)),dtype=int)
        labels_temp=np.append(labels_temp,np.ones((1,(len(noise_points)-len(points))),dtype=int))
        
        labels=np.append(labels,labels_temp)
        estimate_normals(pcd,KDTreeSearchParamHybrid(radius=0.01,max_nn=30))

        source=compute_fpfh_feature(pcd,KDTreeSearchParamHybrid(radius=0.05,max_nn=50))
        feat=source.data
        feat=feat.T

        feat=feat/np.max(feat)
        features.append(feat)
        i+=1

    X_train=np.concatenate(features,axis=0)
    y_train=labels.astype(int)

    return X_train,y_train


def voxel_occupancy_features(point_cloud_path,n_X=8,n_Y=8,n_Z=8):

    
    """
    Returns the voxel occupancy feature 
    for the point cloud
    """
    
    cloud = pyntcloud.PyntCloud.from_file(point_cloud_path)
    voxel_grid_cloud= cloud.add_structure("voxelgrid", n_x=n_X, n_y=n_Y, n_z=n_Z)
    voxelgrid = cloud.structures[voxel_grid_cloud]

    density_feature_vector = voxelgrid.get_feature_vector(mode="density").reshape(-1)

    return density_feature_vector


def feature_extraction_with_voxel_occupancy(directory,voxel_size,num_of_noise_points=1000):
    
    """
    Method to extract features along with voxel occupancy for 
    global context
    """

    labels=[]
    features=[]

    if os.path.isdir("Noisy") is False:
        os.mkdir("Noisy")

    print("Adding noise to point clouds and extracting features (pointwise and voxel occupancy")
    
    for file in tqdm(os.listdir(directory)):

        pcd = read_point_cloud(os.path.join(directory,file))
        points = np.asarray(pcd.points)
        noise_points = reg_noise(points,num_of_noise_points)

        pcd.points=Vector3dVector(noise_points)

        write_point_cloud(os.path.join("Noisy",file)+'_noisy.pcd',pcd)

        estimate_normals(pcd,KDTreeSearchParamHybrid(radius=0.01,max_nn=30))

        labels_temp = np.zeros((1,len(points)),dtype=int)
        labels_temp = np.append(labels_temp,np.ones((1,num_of_noise_points),dtype=int))
        labels = np.append(labels,labels_temp)

        source = compute_fpfh_feature(pcd,KDTreeSearchParamHybrid(radius=0.05,max_nn=50))
        feature = source.data.T
        feature = feature/np.max(feature)

        voxel_features=voxel_occupancy_features(os.path.join(directory,file),voxel_size[0],voxel_size[1],voxel_size[2])

        for item in feature:

            features.append(np.append(item,voxel_features,axis=0))


    X_train=np.array(features)
    y_train=labels.astype(int)
    return X_train,y_train

def denoise_point_cloud(model_file,directory,save_dir,feature_type):

    """
    Method for performing inference and denoising the point clouds
    """

    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model = load_model(model_file)
    files = os.listdir(directory)
    
    print("Denoising the point clouds...")
    
    for file in tqdm(files):
        
        point_cloud = read_point_cloud(os.path.join(directory,file))
        feature_ext = inference_feature_extraction(os.path.join(directory,file),feature_type)
        points = np.asarray(point_cloud.points)
                
        predicted_pts = model.predict(feature_ext)
        predicted_pts =(predicted_pts>0.5).astype(int)
        
        points_copy = [value for value,pred in zip(points,predicted_pts) if pred == 0]
        
        point_cloud.points  = Vector3dVector(points_copy)
        
        write_point_cloud(os.path.join(save_dir,file),point_cloud)

