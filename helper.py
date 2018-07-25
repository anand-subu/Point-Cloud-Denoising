#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:52:19 2018

@author: anand
"""

import numpy as np
from sklearn.preprocessing import Normalizer
#import pydriver
import random
import os 
import keras
import argparse
import open3d
from open3d import *
from scipy.cluster.vq import *



from sklearn.preprocessing import StandardScaler

def reg_noise(pts):
    i=0
    noise=np.zeros((2000,3))
    x,y,z=pts.min(axis=0)
    x_max,y_max,z_max=pts.max(axis=0)
    for i in range(2000):
        a=random.uniform( x, x_max )
        b=random.uniform( y, y_max )
        c=random.uniform( z, z_max )
        noise[i][0]=a
        noise[i][1]=b
        noise[i][2]=c
    return noise
        
def feat_ext(thisdir):
    
    feature=[]
    feats=[]
    labels=[]
    for f in os.listdir(thisdir):
    
    
        pcd=read_point_cloud(os.path.join(thisdir,f))
        points=pcd.points
        points=np.asarray(points)
        noise=reg_noise(points)
        points=np.append(points,noise,axis=0)
        points_temp=points
        
        pcd.points=Vector3dVector(points)

        write_point_cloud(os.path.join("Noisy",f)+'_noisy.pcd',pcd)

        
        labels_temp=np.zeros((1,(len(points)-len(noise))),dtype=int)
        labels_temp=np.append(labels_temp,np.ones((1,len(noise)),dtype=int))
        labels=np.append(labels,labels_temp)
        
        estimate_normals(pcd,KDTreeSearchParamHybrid(radius=0.01,max_nn=30))

        source=compute_fpfh_feature(pcd,KDTreeSearchParamHybrid(radius=0.05,max_nn=50))
        feat=source.data
        feat=feat.T
        descriptors=feat
        k = 200
        voc, variance = kmeans(np.float32(descriptors), k, 1)

        pc_features = np.zeros((1, k), "float32")
        for i in range(1):
            words, distance = vq(descriptors,voc)
            for w in words:
                pc_features[i][w] += 1

        pc_features=pc_features/np.max(pc_features)
        
        for i in range(len(points_temp)):
            t=feat[i]
            t.resize((1,33))
            if np.max(t)!=0:
                t=t/np.max(t)
                c=np.append(t,pc_features,axis=1)
                feats.append(c)

        
            

    X_train=np.array(feats)
    X_train=X_train[:,0]
    labels=labels.astype(int)
    return X_train,labels

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--path", required=True,help="path to the folders with point clouds for training")
args = vars(ap.parse_args())

X_train,y_train=feat_ext(args["path"])

np.save('X_train',X_train)
np.save('y_train',y_train)