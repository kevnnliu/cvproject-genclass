import tensorflow as tf
from keras import backend as kb

from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU, SpatialDropout2D, Dropout, BatchNormalization, GaussianNoise
from keras.applications import resnet_v2

import math
gsd = 1 / math.sqrt(2 * math.pi)

# BravoNet:
# Residual network using the ResNet152 v2 architecture.
# Training time: XX epochs, ~XX hours
# Trainable parameters: 58,597,704
# Best: ~XX% validation accuracy
def BravoNet():
    model = Sequential()
    model.name = "BravoNet"
    model.add(GaussianNoise(gsd, input_shape=(64, 64, 3)))

    base_model = resnet_v2.ResNet152V2(weights=None, include_top=False, pooling="avg")
    model.add(base_model)

    model.add(Dense(200, activation="softmax"))

    return model

# AlphaNet:
# Simple convolutional network with minor improvements.
# Training time: XX epochs, ~XX hours
# Trainable parameters: 23,000,776
# Best: ~XX% validation accuracy
def AlphaNet():
    model = Sequential()
    model.name = "AlphaNet"
    model.add(GaussianNoise(gsd, input_shape=(64, 64, 3)))

    model.add(Conv2D(32, kernel_size=7, padding="same", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.4))

    model.add(Conv2D(64, kernel_size=5, padding="same", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.4))

    model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.4))

    model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(512, kernel_size=3, padding="same", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.4))

    model.add(Flatten())
    
    model.add(Dense(2048, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(2048, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(200, activation="softmax"))

    return model
