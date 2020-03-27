import tensorflow as tf
from keras import backend as kb

from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU, SpatialDropout2D, Dropout, BatchNormalization
from keras.applications import vgg16, resnet_v2

# BravoNet:
# Transfer learning using VGG16 as base model.
def BravoNet():
    base_model = vgg16.VGG16(weights=None, include_top=False, input_shape=(64, 64, 3))

    model = Sequential()
    model.add(base_model)
    model.name = 'BravoNet'

    model.add(Flatten())
    
    model.add(Dense(2048, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(2048, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(200, activation='softmax'))

    return model

# AlphaNet:
# Naive approach using basic convolutional blocks.
# Achieved 39.4% validation accuracy after 232 epochs.
def AlphaNet():
    model = Sequential()
    model.name = 'AlphaNet'

    model.add(Conv2D(32, kernel_size=7, padding='same', kernel_initializer='glorot_normal', input_shape=(64, 64, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.4))

    model.add(Conv2D(64, kernel_size=5, padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.4))

    model.add(Conv2D(128, kernel_size=3, padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.4))

    model.add(Conv2D(256, kernel_size=3, padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(512, kernel_size=3, padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.4))

    model.add(Flatten())
    
    model.add(Dense(2048, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(2048, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(200, activation='softmax'))

    return model
