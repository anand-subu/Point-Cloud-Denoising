#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:56:53 2018

@author: anand
"""

import numpy as np
import random
import os 
import argparse
from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn.model_selection import train_test_split
from logging import StreamHandler, INFO, getLogger
import tensorflow as tf
import tensorflow.contrib.learn as learn
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
import tensorflow.contrib.metrics as tfmetrics
from open3d import *

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--test_pointcloud", required=True,help="path to a testpoint cloud for visualizing results")
args = vars(ap.parse_args())


X_train=np.load('X_train.npy')
y_train=np.load('y_train.npy')

X_train=np.float32(X_train)

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=np.random.seed(7))
#X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.2,random_state=np.random.seed(7))

y_train=np.int32(y_train)

y_val=np.int32(y_val)
y_train=np.reshape(y_train,(len(y_train),))

y_val=np.reshape(y_val,(len(y_val),))

logger = getLogger('tensorflow')
logger.removeHandler(logger.handlers[0])

logger.setLevel(INFO)

class DebugFileHandler(StreamHandler):
    def __init__(self):
        StreamHandler.__init__(self)

    def emit(self, record):
        if not record.levelno == INFO:
            return
        StreamHandler.emit(self, record)

logger.addHandler(DebugFileHandler())


def ANN(features, labels,mode):
    labels = tf.one_hot(tf.cast(labels, tf.int32), 2, 1, 0)

    dense1 = tf.layers.dense(features, 128, activation=tf.nn.relu, name='fc1')
    dense2 = tf.layers.dense(dense1, 64, activation=tf.nn.relu, name='fc2')
    dense3 = tf.layers.dense(dense2, 64, activation=tf.nn.relu, name='fc3')
    dropout1=tf.layers.dropout(dense3, rate=0.5)
    dense4 = tf.layers.dense(dropout1, 32, activation=tf.nn.relu, name='fc4')
    logits = tf.layers.dense(dense4, 2,activation=None, name='out')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss, tf.contrib.framework.get_global_step())
    
    
    return tf.argmax(logits, 1), loss, train_op


classifier = learn.Estimator(model_fn=ANN,
                             model_dir="./output",
                             config=learn.RunConfig(save_checkpoints_secs=10))


validation_monitor = learn.monitors.ValidationMonitor(
    x=X_val,
    y=y_val,
    metrics={'accuracy': MetricSpec(tfmetrics.streaming_accuracy)},
    every_n_steps=500)

classifier.fit(x=X_train,
               y=y_train,
               batch_size=16,
               steps=10000,
               monitors=[validation_monitor])



#score = classifier.evaluate(x=X_test, y=y_test,metrics={'accuracy': MetricSpec(tfmetrics.streaming_accuracy)})
#print('Accuracy: {0:f}'.format(score['accuracy']))






i=0
directory="Noisy"
noiseless=[]

for f in os.listdir(directory):
    
    pcd=read_point_cloud(os.path.join(directory,f))
    feat=X_train[i]
    predictions = np.asarray(list(classifier.predict(x=feat)))
    points=pcd.points
    points=np.array(points)
    
    for i in range(len(points)):
        if predictions[i]==0:
            noiseless.append(points[i])
    
    noiseless=np.asarray(noiseless)
    
    noiseless_pcd=PointCloud()
    
    noiseless_pcd.points=Vector3dVector(noiseless)
    
    draw_geometries([pcd]) #noisy point cloud
    draw_geometries([noiseless_pcd]) #noiseless point cloud
    
    
    
    
            
            
            
            
    
    
    
    
    
    
        

