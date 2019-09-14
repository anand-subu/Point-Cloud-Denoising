import argparse
from model import define_model
from helper import *

if __name__ == "__main__": 

    
    parser = argparse.ArgumentParser(description='Arguements for training the model')
    
    parser.add_argument('--train_with_local_features', type=bool, nargs='+',
                    help='extract only local features for training')
    parser.add_argument('--train_with_global_features',type=bool,
                    help='extract only global features')
    parser.add_argument('--voxel_size',type=str,nargs='?',
                    help='optional arguement, needed if training with global features\
                          Of the format [x,y,z] corresponding to voxel dimenstions',required=False)
    parser.add_argument('--train_directory', type=str,
                    help='Directory with pointclouds for training')
    parser.add_argument('--test_directory', type=str,
                    help='Directory with pointclouds for testing')

    args = parser.parse_args()

    if args.train_with_global_features:
        
        voxel_size = args.voxel_size.split(",")
        voxel_size = [int(x) for x in voxel_size]
        
        input_size = (voxel_size[0] * voxel_size[1] * voxel_size[2])+33
        model = define_model("global",input_size)
        
        X_train,y_train=feature_extraction_with_voxel_occupancy(args.train_directory,voxel_size,1000)

        X_test,y_test=feature_extraction_with_voxel_occupancy(args.test_directory,voxel_size,1000)
        
    if args.train_with_local_features:
        
        model=define_model("local",input_size=33)
        
        X_train,y_train=feature_extraction(args.train_directory)

        X_test,y_test=feature_extraction(args.test_directory)

    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(X_train, y_train,
                        epochs=1,
                        batch_size=64,
                        shuffle=True,
                        validation_split=0.2)
    model.save("final_model.h5")
