import tensorflow as tf
from keras import backend as kb

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU, SpatialDropout2D, Dropout, BatchNormalization

def AlphaNet():
    model = Sequential()
    model.name = 'AlphaNet'

    model.add(Conv2D(32, kernel_size=7, padding='same', input_shape=(64, 64, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.3))

    model.add(Conv2D(64, kernel_size=5, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(64, kernel_size=5, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.3))

    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.3))

    model.add(Flatten())
    
    model.add(Dense(1536))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))

    model.add(Dense(200, activation='softmax'))

    return model
