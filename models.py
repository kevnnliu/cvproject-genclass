import tensorflow as tf
import time

from keras.models import Model, Sequential
from keras.layers import (Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
                          Flatten, LeakyReLU, SpatialDropout2D, Dropout,
                          BatchNormalization, Input, Concatenate)

from classification_models.resnet import ResNet18, ResNet34


# AlphaStack:
# Integrated stacking ensemble using AlphaNet and BravoNet as members.
def AlphaStack(input_shape, alphas, m_train=False, version="", quiet=False):
    # Freeze member layers
    for alpha in alphas:
        for layer in alpha.layers:
            layer.trainable = m_train

    input_layer = Input(shape=input_shape)

    alpha_output = [alpha(input_layer) for alpha in alphas]

    # Merge outputs
    merge = Concatenate()(alpha_output)

    hidden_1 = Dense(512, kernel_initializer="glorot_normal")(merge)
    relu_1 = LeakyReLU(alpha=0.1)(hidden_1)

    hidden_2 = Dense(512, kernel_initializer="glorot_normal")(relu_1)
    relu_2 = LeakyReLU(alpha=0.1)(hidden_2)

    hidden_3 = Dense(512, kernel_initializer="glorot_normal")(relu_2)
    relu_3 = LeakyReLU(alpha=0.1)(hidden_3)

    final_output = Dense(200, activation="softmax")(relu_3)

    model = Model(inputs=input_layer, outputs=final_output)
    model.name = append_version("AlphaStack", version)

    if not quiet:
        model.summary()

    return model


# BravoNet:
# ResNet18 architecture.
def BravoNet(input_shape, version="", weights=None, quiet=False):
    model = Sequential()
    model.name = append_version("BravoNet", version)

    base_model = ResNet18(weights=weights,
                          include_top=False,
                          classes=200,
                          input_shape=input_shape)

    if not quiet:
        base_model.summary()

    model.add(base_model)
    model.add(GlobalAveragePooling2D())

    model.add(Dense(200, activation="softmax"))

    if not quiet:
        model.summary()

    return model


# AlphaNet:
# Standard convolution.
def AlphaNet(input_shape, version="", quiet=False):
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

    if not quiet:
        model.summary()

    return model


def append_version(name, version):
    suffix = "" if version == "" else "_" + version
    return name + suffix
