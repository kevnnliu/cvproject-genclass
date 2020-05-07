import tensorflow as tf
from keras import backend as kb
import autokeras as ak
import time

from keras.models import Model, Sequential
from keras.layers import (Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
                          Flatten, LeakyReLU, SpatialDropout2D, Dropout,
                          BatchNormalization, Lambda)
from keras.applications import (resnet_v2, xception, nasnet, mobilenet_v2,
                                densenet, inception_v3, inception_resnet_v2)

from classification_models.resnet import ResNet18


# JuliNet:
# Inception-residual network using the Inception-ResNet V2 architecture.
def JuliNet(version=""):
    model = Sequential()
    model.name = append_version("JuliNet", version)

    model.add(resize_gaussian())

    base_model = inception_resnet_v2.InceptionResNetV2(include_top=False,
                                                       weights=None,
                                                       pooling="avg",
                                                       classes=200)

    base_model.summary()

    model.add(base_model)

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


# IndiaNet:
# Inception network using the Inception V3 architecture.
def IndiaNet(version=""):
    model = Sequential()
    model.name = append_version("IndiaNet", version)

    model.add(resize_gaussian())

    base_model = inception_v3.InceptionV3(include_top=False,
                                          weights=None,
                                          pooling="avg",
                                          classes=200)

    base_model.summary()

    model.add(base_model)

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


# HotelNet:
# Densely connected convolutional network using the DenseNet121 architecture.
def HotelNet(version=""):
    model = Sequential()
    model.name = append_version("HotelNet", version)

    model.add(resize_gaussian())

    base_model = densenet.DenseNet121(include_top=False,
                                      weights=None,
                                      pooling="avg",
                                      classes=200)

    base_model.summary()

    model.add(base_model)

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


# GolfNet:
# Efficient convolutional network using the MobileNetV2 architecture.
def GolfNet(version=""):
    model = Sequential()
    model.name = append_version("GolfNet", version)

    model.add(resize_gaussian())

    base_model = mobilenet_v2.MobileNetV2(include_top=False,
                                          weights=None,
                                          pooling="avg",
                                          classes=200)

    base_model.summary()

    model.add(base_model)

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


# FoxNet:
# Neural architecture search using the NASNetMobile architecture.
def FoxNet(version=""):
    model = Sequential()
    model.name = append_version("FoxNet", version)

    model.add(resize_gaussian())

    base_model = nasnet.NASNetMobile(include_top=False,
                                     weights=None,
                                     pooling="avg",
                                     classes=200)

    base_model.summary()

    model.add(base_model)

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


# EchoNet:
# Neural architecture search using the NASNetLarge architecture.
def EchoNet(version=""):
    model = Sequential()
    model.name = append_version("EchoNet", version)

    model.add(resize_gaussian())

    base_model = nasnet.NASNetLarge(include_top=False,
                                    weights=None,
                                    pooling="avg",
                                    classes=200)

    base_model.summary()

    model.add(base_model)

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


# DeltaNet:
# Depthwise separable convolutional network using the Xception V1 architecture.
def DeltaNet(version=""):
    model = Sequential()
    model.name = append_version("DeltaNet", version)

    model.add(resize_gaussian())

    base_model = xception.Xception(include_top=False,
                                   weights=None,
                                   pooling="avg",
                                   classes=200)

    base_model.summary()

    model.add(base_model)

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


# CharlieNet:
# Neural architecture search using the AutoKeras library.
def CharlieNet(data, overwrite, version=""):
    model = Sequential()
    model.name = append_version("CharlieNet", version)

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]

    node_input = ak.ImageInput(shape=(64, 64, 3))
    node_head = ak.ClassificationHead(num_classes=200)

    automodel = ak.AutoModel(inputs=node_input,
                             outputs=node_head,
                             directory="models\\" + model.name,
                             objective="val_accuracy",
                             tuner="bayesian",
                             overwrite=overwrite)

    print("Began neural architecture search.")
    start_time = time.time()

    automodel.fit(x=X_train,
                  y=y_train,
                  epochs=50,
                  validation_data=(X_val, y_val),
                  verbose=2)

    print("Completed neural architecture search in %fs." %
          (time.time() - start_time))

    nas_model = automodel.export_model()
    nas_model.summary()

    model.add(nas_model)
    model.summary()

    return model


# BravoNet:
# Residual network using the ResNet18 architecture.
def BravoNet(version=""):
    model = Sequential()
    model.name = append_version("BravoNet", version)

    model.add(resize_gaussian())

    base_model = ResNet18(weights=None,
                          include_top=False,
                          classes=200,
                          input_shape=(64, 64, 3))

    model.add(base_model)
    model.add(GlobalAveragePooling2D())

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


# AlphaNet:
# Simple convolutional network with minor improvements.
def AlphaNet(version=""):
    model = Sequential()
    model.name = append_version("AlphaNet", version)

    model.add(resize_gaussian())

    model.add(
        Conv2D(32,
               kernel_size=7,
               padding="same",
               kernel_initializer="glorot_normal",
               input_shape=(64, 64, 3)))
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

    model.add(Dense(2048, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(2048, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


def append_version(name, version):
    suffix = "" if version == "" else "_" + version
    return name + suffix


def resize_gaussian(factor=2):
    new_dim = factor * 64
    return Lambda(lambda batch: tf.image.resize(
        images=batch, size=(new_dim, new_dim), method="gaussian"),
                  input_shape=(64, 64, 3))
