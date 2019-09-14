import pyntcloud
import numpy as np
import argparse
import os 
from open3d import *
from helper import *
from tqdm import tqdm

        

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Arguements for training the model')
    
    parser.add_argument('--path_to_model', type=str,
                    help='Path to saved model file')
    parser.add_argument('--inference_dir', type=str,
                    help='Directory with pointclouds for inference')
    parser.add_argument('--save_dir', type=str,
                    help='Directory for saving the denoised pointclouds')
    parser.add_argument('--feature_type', type=str,
                    help='Type of feature extraction (global or local')
    args = parser.parse_args()

    denoise_point_cloud(args.path_to_model,args.inference_dir,args.save_dir,args.feature_type)    
    
    