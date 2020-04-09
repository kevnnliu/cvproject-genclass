import tensorflow as tf
from keras import backend as kb

from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU, SpatialDropout2D, Dropout, BatchNormalization, GaussianNoise
from keras.applications import resnet_v2

# BravoNet:
# Residual network using the ResNet50 v2 architecture.
def BravoNet(version):
    model = Sequential()
    model.name = "BravoNet" + version

    base_model = resnet_v2.ResNet50V2(weights='imagenet', include_top=False, pooling="avg", input_shape=(64, 64, 3))
    model.add(base_model)

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model

# AlphaNet:
# Simple convolutional network with minor improvements.
def AlphaNet(version):
    model = Sequential()
    model.name = "AlphaNet" + version

    model.add(Conv2D(32, kernel_size=7, padding="same", kernel_initializer="glorot_normal", input_shape=(64, 64, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(Conv2D(64, kernel_size=5, padding="same", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(Conv2D(128, kernel_size=3, padding="same", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(512, kernel_size=3, padding="same", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())
    
    model.add(Dense(2048, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))

    model.add(Dense(2048, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model
