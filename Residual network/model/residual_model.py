import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block
    Input activation has the same dimension as the output activation
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    # Save the input value as residual
    X_shortcut = X
    
    # First component of main path, Xavier initializer
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1,1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=42))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=42))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=42))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Add shortcut to main path as residual, then pass it through RELU
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block
    Input dimension does not match with output dimension
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    # Save the input value as residual
    X_shortcut = X

    # First component of main path 
    X = Conv2D(F1, (1, 1), strides=(s,s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=42))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    

    # Second component of main path
    X = Conv2D(F2, (f, f), strides=(1,1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=42))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, (1,1), strides=(1,1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=42))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Resize residual input dimension to be the same as the last output activation
    X_shortcut = Conv2D(F3, (1,1), strides=(s,s), name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=42))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Add shortcut to main path as residual, then pass it through RELU
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet_50(input_shape=(64, 64, 3), classes=6, zero_padding=3):
    """
    Implementation of the ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Returns:
    model -- a Model() instance in Keras
    """
    
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((zero_padding, zero_padding))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL as feature extractor
    X = AveragePooling2D(pool_size=(2,2))(X)
    

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=42))(X)
    
    model = Model(inputs = X_input, outputs = X, name='ResNet_50')

    return model
