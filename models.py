import tensorflow as tf
import time

from keras.models import Model, Sequential
from keras.layers import (Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
						  Flatten, LeakyReLU, SpatialDropout2D, Dropout,
						  BatchNormalization, Input, Concatenate)
from keras.applications import (mobilenet_v2, densenet, vgg16)

from classification_models.resnet import ResNet18, ResNet34


# AlphaBravo:
# Integrated stacking ensemble using AlphaNet and BravoNet as members.
def AlphaBravo(alpha: Model, bravo: Model, version=""):
	# Freeze member layers
	for layer in alpha.layers:
		layer.trainable = False
	for layer in bravo.layers:
		layer.trainable = False

	input_layer = Input(shape=(64, 64, 3))

	alpha_output = alpha(input_layer)
	bravo_output = bravo(input_layer)

	# Merge outputs
	merge = Concatenate()([alpha_output, bravo_output])

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

	final_output = Dense(200, activation="softmax")(dp_3)

	model = Model(inputs=input_layer, outputs=final_output)
	model.name = append_version("AlphaBravo", version)

	return model


# BravoNet:
# Residual network using the ResNet18 architecture.
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
# Simple convolutional network with minor improvements.
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
