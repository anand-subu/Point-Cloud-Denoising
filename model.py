from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization

def define_model(feature_type,input_size):
    
    """ 
    No frills MultiLayer Perceptron for training the denoising network. 
    Defining the model input based on the type of features extracted 
    for training  the network
    """
   
    if feature_type == "global":
        input_layer = Input(shape=(input_size,))
        x = Dense(256, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        output = Dense(1, activation='sigmoid')(x)
    
        model = Model(input_layer, output)
        return model        
    

    if feature_type == "local":

        input_layer = Input(shape=(33,))
        x = Dense(32, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)        
        x = Dropout(0.5)(x)
        x = Dense(5, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)
    
        model = Model(input_layer, output)
    
    return model






