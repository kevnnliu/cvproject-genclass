import tensorflow as tf
from keras import backend as kb
import autokeras as ak
import time

from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU, SpatialDropout2D, Dropout, BatchNormalization, GaussianNoise
from keras.applications import resnet_v2


# CharlieNet:
# Neural architecture search using AutoKeras.
def CharlieNet(data, version=""):
    model = Sequential()
    model.name = version("CharlieNet", version)

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]

    node_input = ak.ImageInput(shape=(64, 64, 3))
    node_head = ak.ClassificationHead(num_classes=200)

    automodel = ak.AutoModel(inputs=node_input,
                             outputs=node_head,
                             directory="models\\" + model.name,
                             objective="val_accuracy",
                             tuner="hyperband",
                             overwrite=True)

    print("Began neural architecture search.")
    start_time = time.time()

    automodel.fit(x=X_train,
                  y=y_train,
                  epochs=50,
                  validation_data=(X_val, y_val))

    print("Completed neural architecture search in %fs." % (time.time() - start_time))

    nas_model = automodel.export_model()
    nas_model.summary()

    model.add(nas_model)
    model.summary()

    return model


# BravoNet:
# Residual network using the ResNet50 v2 architecture.
def BravoNet(version=""):
    model = Sequential()
    model.name = version("BravoNet", version)

    base_model = resnet_v2.ResNet50V2(weights=None,
                                      include_top=False,
                                      pooling="avg",
                                      input_shape=(64, 64, 3))
    model.add(base_model)

    model.add(Dense(200, activation="softmax"))

    model.summary()

    return model


# AlphaNet:
# Simple convolutional network with minor improvements.
def AlphaNet(version=""):
    model = Sequential()
    model.name = version("AlphaNet", version)

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


def version(name, version):
    suffix = "" if version == "" else "_" + version
    return name + suffix
