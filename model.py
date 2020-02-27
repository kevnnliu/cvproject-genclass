import tensorflow as tf
from keras import backend as kb

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU

def AlphaNet():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=(64, 64, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(32, kernel_size=3, padding='valid'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, kernel_size=3, padding='valid'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, kernel_size=3, padding='valid'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D())

    model.add(Flatten())
    
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.3))

    model.add(Dense(200, activation='softmax'))

    return model
