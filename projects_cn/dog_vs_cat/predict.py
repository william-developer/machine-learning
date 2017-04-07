from keras.layers import Input, merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from DataUtil import load
from keras.utils import np_utils, generic_utils

from keras import backend as K

def stem(input):
    # Input Shape is 299 x 299 x 3 
    c = Convolution2D(32, 3, 3, activation='relu', subsample=(2, 2))(input)
    c = Convolution2D(32, 3, 3, activation='relu', )(c)
    c = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(c)

    c1 = MaxPooling2D((3, 3), strides=(2, 2))(c)
    c2 = Convolution2D(96, 3, 3, activation='relu', subsample=(2, 2))(c)

    m = merge([c1, c2], mode='concat', concat_axis=-1)

    c1 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c1 = Convolution2D(96, 3, 3, activation='relu', )(c1)

    c2 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c2 = Convolution2D(64, 7, 1, activation='relu', border_mode='same')(c2)
    c2 = Convolution2D(64, 1, 7, activation='relu', border_mode='same')(c2)
    c2 = Convolution2D(96, 3, 3, activation='relu', border_mode='valid')(c2)

    m2 = merge([c1, c2], mode='concat', concat_axis=-1)

    p1 = MaxPooling2D((3, 3), strides=(2, 2), )(m2)
    p2 = Convolution2D(192, 3, 3, activation='relu', subsample=(2, 2))(m2)

    m3 = merge([p1, p2], mode='concat', concat_axis=-1)
    m3 = BatchNormalization(axis=-1)(m3)
    m3 = Activation('relu')(m3)
    return m3

def inception_resnet_A(input):
    init = input

    ir1 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(ir2)

    ir3 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir3 = Convolution2D(48, 3, 3, activation='relu', border_mode='same')(ir3)
    ir3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(ir3)

    ir_merge = merge([ir1, ir2, ir3], concat_axis=-1, mode='concat')

    ir_conv = Convolution2D(384, 1, 1, activation='linear', border_mode='same')(ir_merge)
    ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = BatchNormalization(axis=-1)(out)
    out = Activation("relu")(out)
    return out

def inception_resnet_B(input):
    init = input

    ir1 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(160, 1, 7, activation='relu', border_mode='same')(ir2)
    ir2 = Convolution2D(192, 7, 1, activation='relu', border_mode='same')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=-1)

    ir_conv = Convolution2D(1152, 1, 1, activation='linear', border_mode='same')(ir_merge)
    ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = BatchNormalization(axis=-1)(out)
    out = Activation("relu")(out)
    return out

def inception_resnet_C(input):
    init = input

    ir1 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(224, 1, 3, activation='relu', border_mode='same')(ir2)
    ir2 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=-1)

    ir_conv = Convolution2D(2144, 1, 1, activation='linear', border_mode='same')(ir_merge)
    ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = BatchNormalization(axis=-1)(out)
    out = Activation("relu")(out)
    return out


def reduction_A(input, k=192, l=224, m=256, n=384):
    r1 = MaxPooling2D((3,3), strides=(2,2))(input)

    r2 = Convolution2D(n, 3, 3, activation='relu', subsample=(2,2))(input)

    r3 = Convolution2D(k, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(l, 3, 3, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(m, 3, 3, activation='relu', subsample=(2,2))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=-1)
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m


def reduction_B(input):
    r1 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(input)

    r2 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r2 = Convolution2D(384, 3, 3, activation='relu', subsample=(2,2))(r2)

    r3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(288, 3, 3, activation='relu', subsample=(2, 2))(r3)

    r4 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r4 = Convolution2D(288, 3, 3, activation='relu', border_mode='same')(r4)
    r4 = Convolution2D(320, 3, 3, activation='relu', subsample=(2, 2))(r4)

    m = merge([r1, r2, r3, r4], concat_axis=-1, mode='concat')
    m = BatchNormalization(axis=-1)(m)
    m = Activation('relu')(m)
    return m

def create_inception_resnet_v2(nb_classes=2):
    init = Input((299, 299, 3))

    x = stem(init)

    for i in range(5):
        x = inception_resnet_A(x)

    x = reduction_A(x, k=256, l=256, m=384, n=384)

    for i in range(10):
        x = inception_resnet_B(x)

    x = reduction_B(x)

    for i in range(5):
        x = inception_resnet_C(x)

    x = AveragePooling2D((8,8))(x)

    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    out = Dense(output_dim=nb_classes, activation='softmax')(x)

    model = Model(init, output=out, name='Inception-Resnet-v2')
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
    return model

if __name__ == "__main__":
    model = create_inception_resnet_v2()
    data,label = load('data')	
    label=np_utils.to_categorical(label, 2)
    
    model.fit(data, label, batch_size=32,nb_epoch=10,shuffle=True,verbose=2,validation_split=0.2)
    model.save('tf-cat-dog-weight-normal.h5')
