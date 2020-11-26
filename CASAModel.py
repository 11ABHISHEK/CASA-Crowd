import numpy as np
from keras.applications.vgg16 import VGG16
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,add,UpSampling2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Concatenate
from keras.initializers import RandomNormal

def CASAModel(input_shape=(None,None,3)):
    X_input=Input(input_shape)
    dilated_conv_kernel_initializer = RandomNormal(stddev=0.01)
# First block
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(X_input)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
# Second block
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
# Third block
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
# Fourth block
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    X0 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
#fifth block
    X1 = Conv2D(256, (3, 3), strides = (1, 1),padding='same',dilation_rate=2, activation='relu')(X0)
#     X = ZeroPadding2D((2,2))(X0)
    X2 = Conv2D(256, (5, 5), strides = (1, 1),padding='same',dilation_rate=3, activation='relu')(X1)
#     X = ZeroPadding2D((3,3))(X0)
    X3 = Conv2D(256, (7, 7), strides = (1, 1),padding='same',dilation_rate=4, activation='relu')(X2)
    merge=Concatenate()([X1,X2])
    X0=Concatenate()([X3,merge])
    X = Conv2D(256, (1,1), strides = (1, 1), activation='relu',
               kernel_initializer=dilated_conv_kernel_initializer)(X0)
    #sixth block
    X1 = Conv2D(128, (3, 3), strides = (1, 1),padding='same',dilation_rate=2, activation='relu')(X)
#     X = ZeroPadding2D((2,2))(X0)
    X2 = Conv2D(128, (5, 5), strides = (1, 1),padding='same',dilation_rate=3, activation='relu')(X1)
#     X = ZeroPadding2D((3,3))(X0)
    X3 = Conv2D(128, (7, 7), strides = (1, 1),padding='same',dilation_rate=4, activation='relu')(X2)
    merge=Concatenate()([X1,X2])
    X0=Concatenate()([X3,merge])
    X = Conv2D(128, (1,1), strides = (1, 1), activation='relu',
               kernel_initializer=dilated_conv_kernel_initializer)(X0)
    #seventh block
    X1 = Conv2D(64, (3, 3), strides = (1, 1),padding='same',dilation_rate=2,  activation='relu')(X)
#     X = ZeroPadding2D((2,2))(X0)
    X2 = Conv2D(64, (5, 5), strides = (1, 1),padding='same',dilation_rate=3,  activation='relu')(X1)
#     X = ZeroPadding2D((3,3))(X0)
    X3 = Conv2D(64, (7, 7), strides = (1, 1),padding='same',dilation_rate=4, activation='relu')(X2)
    merge=Concatenate()([X1,X2])
    X=Concatenate()([X3,merge])
    X = Conv2D(64, (3,3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(X)
    X = Conv2D(1, 1, strides=(1, 1), padding='same', activation='relu', kernel_initializer=dilated_conv_kernel_initializer)(X)
    
    model=Model(inputs=X_input, outputs=X,name='CASAModel')
    
    front_end = VGG16(weights='imagenet', include_top=False)

    weights_front_end = []
    for layer in front_end.layers:
        if 'conv' in layer.name:
            weights_front_end.append(layer.get_weights())
    counter_conv = 0
    for i in range(len(front_end.layers)):
        if counter_conv >= 10:
            break
        if 'conv' in model.layers[i].name:
            model.layers[i].set_weights(weights_front_end[counter_conv])
            counter_conv += 1
    return model