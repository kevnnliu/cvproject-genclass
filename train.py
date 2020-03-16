import tensorflow as tf
from keras.models import load_model
from keras.callbacks import callbacks
import sys
import models

from tools import load_data
from os import path, remove
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

# Use these to test check for gpu availability
tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')

def train(restore=False, epochs=40, model=None, optim='adam', batch_size=16, datagen=None, data=None, cb_list=None):
    model_path = 'models/' + model.name + '.h5'

    X_train, y_train = data['train']
    X_val, y_val = data['val']

    if path.exists(model_path):
        if restore:
            model = load_model(model_path)
        else:
            remove(model_path)

    model.summary()
    model.compile(optim, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), 
                                    steps_per_epoch=len(X_train) / batch_size, 
                                    epochs=epochs, verbose=1, callbacks=cb_list, 
                                    validation_data=datagen.flow(X_val, y_val, batch_size=batch_size),
                                    validation_steps=len(X_val) / batch_size)

    model.save(model_path)

    return history
