import tensorflow as tf
from keras.models import load_model
from keras.callbacks import callbacks
import sys
import models

from tools import load_data
from os import path, remove
import numpy as np
import matplotlib.pyplot as plt

# Use these to test check for gpu availability
tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')


def train(delete_old=False, num_epochs=20, model_name='AlphaNet', optim='adam'):
    model_path = 'models/' + model_name + '.h5'

    X_train, y_train = load_data()
    X_val, y_val = load_data('val')

    model = getattr(models, model_name)()

    if path.exists(model_path):
        if delete_old:
            remove(model_path)
        else:
            model = load_model(model_path)

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.summary()
    model.compile(optim, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), verbose=2, callbacks=[early_stop])

    model.save(model_path)

    return history
