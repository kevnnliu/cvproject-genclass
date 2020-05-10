import tensorflow as tf
from keras import backend as kb
import time

from keras.models import Model, Sequential
from keras.layers import (Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
                          Flatten, LeakyReLU, SpatialDropout2D, Dropout,
                          BatchNormalization, Lambda)
from keras.applications import (mobilenet_v2, densenet, vgg16)

from classification_models.resnet import ResNet18


# HotelNet:
# Densely connected convolutional network using the DenseNet121 architecture.
def HotelNet(input_shape, version="", weights=None):
    model = Sequential()
    model.name = append_version("HotelNet", version)

    base_model = densenet.DenseNet121(include_top=True,
                                      weights=weights,
                                      pooling="avg",
                                      classes=200,
                                      input_shape=input_shape)

    base_model.summary()

    model.add(base_model)

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


# GolfNet:
# Efficient convolutional network using the MobileNetV2 architecture.
def GolfNet(input_shape, version="", weights=None):
    model = Sequential()
    model.name = append_version("GolfNet", version)

    base_model = mobilenet_v2.MobileNetV2(include_top=True,
                                          weights=weights,
                                          pooling="avg",
                                          classes=200,
                                          input_shape=input_shape)

    base_model.summary()

    model.add(base_model)

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


# BravoNet:
# Residual network using the ResNet18 architecture.
def BravoNet(input_shape, version="", weights=None):
    model = Sequential()
    model.name = append_version("BravoNet", version)

    base_model = ResNet18(weights=weights,
                          include_top=True,
                          classes=200,
                          input_shape=input_shape)

    model.add(base_model)
    model.add(GlobalAveragePooling2D())

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


# AlphaNet:
# Simple convolutional network with minor improvements.
def AlphaNet(input_shape, version=""):
    model = Sequential()
    model.name = append_version("AlphaNet", version)

    model.add(
        Conv2D(32,
               kernel_size=7,
               padding="same",
               kernel_initializer="glorot_normal",
               input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(
        Conv2D(64,
               kernel_size=5,
               padding="same",
               kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(
        Conv2D(128,
               kernel_size=3,
               padding="same",
               kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(
        Conv2D(256,
               kernel_size=3,
               padding="same",
               kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(
        Conv2D(512,
               kernel_size=3,
               padding="same",
               kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())

    model.add(Dense(1024, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(1024, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


def append_version(name, version):
    suffix = "" if version == "" else "_" + version
    return name + suffix
