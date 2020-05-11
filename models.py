import tensorflow as tf
from keras import backend as kb
import time

from keras.models import Model, Sequential
from keras.layers import (Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
                          Flatten, LeakyReLU, SpatialDropout2D, Dropout,
                          BatchNormalization, Lambda, concatenate)
from keras.applications import (mobilenet_v2, densenet, vgg16)

from classification_models.resnet import ResNet18, ResNet34


# AlphaBravo:
# Integrated stacking ensemble using the models below.
def AlphaBravo(members, version=""):
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            layer.trainable = False
            layer.name = "ensemble_" + str(i + 1) + "_" + layer.name

    ensemble_inputs = [model.input for model in members]
    ensemble_outputs = [model.output for model in members]

    merge = concatenate(ensemble_outputs)

    hidden_1 = Dense(1024, kernel_initializer="glorot_normal")(merge)
    bn_1 = BatchNormalization()(hidden_1)
    relu_1 = LeakyReLU(alpha=0.1)(bn_1)
    dp_1 = Dropout(0.2)(relu_1)

    hidden_2 = Dense(1024, kernel_initializer="glorot_normal")(dp_1)
    bn_2 = BatchNormalization()(hidden_2)
    relu_2 = LeakyReLU(alpha=0.1)(bn_2)
    dp_2 = Dropout(0.2)(relu_2)

    hidden_3 = Dense(1024, kernel_initializer="glorot_normal")(dp_2)
    bn_3 = BatchNormalization()(hidden_3)
    relu_3 = LeakyReLU(alpha=0.1)(bn_3)
    dp_3 = Dropout(0.2)(relu_3)

    output = Dense(200, activation="softmax")(dp_3)

    model = Model(inputs=ensemble_inputs, outputs=output)
    model.name = append_version("AlphaBravo", version)

    return model


# BravoNet:
# Residual network using the ResNet18/ResNet34 architecture.
def BravoNet(input_shape, version="", weights=None, net34=False):
    model = Sequential()
    model.name = append_version("BravoNet", version)

    base_model = ResNet18(weights=weights,
                          include_top=False,
                          classes=200,
                          input_shape=input_shape)

    if net34:
        base_model = ResNet34(weights=weights,
                              include_top=False,
                              classes=200,
                              input_shape=input_shape)

    base_model.summary()

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
