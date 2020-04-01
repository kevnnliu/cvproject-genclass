import tensorflow as tf
from keras import backend as kb

from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU, SpatialDropout2D, Dropout, BatchNormalization
from keras.applications import resnet_v2

# BravoNet:
# Residual network using the ResNet50 v2 architecture with a dropout layer at the end.
# Training time: XX epochs, ~XX hours, ~XXs/epoch, ~XXms/step
# Trainable parameters: 23,929,160
# Best: XX% validation accuracy
def BravoNet():
    model = Sequential()
    model.name = "BravoNet"

    base_model = resnet_v2.ResNet50V2(weights=None, include_top=False, input_shape=(64, 64, 3), pooling="avg")
    model.add(base_model)
    model.add(Dropout(0.5))

    model.add(Dense(200, activation="softmax"))

    return model

# AlphaNet:
# Simple convolutional network with batch normalization, dropout, and max pooling.
# Training time: XX epochs, ~XX hours, ~150s/epoch, ~50ms/step
# Trainable parameters: 23,000,776
# Best: XX% validation accuracy
def AlphaNet():
    model = Sequential()
    model.name = "AlphaNet"

    model.add(Conv2D(32, kernel_size=7, padding="same", kernel_initializer="glorot_normal", input_shape=(64, 64, 3)))
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
