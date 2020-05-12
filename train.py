import tensorflow as tf
from keras.models import load_model
import models
import random
import numpy as np
import keras.metrics

from os import path, remove
import matplotlib.pyplot as plt
import functools

# Use these to test check for gpu availability
tf.test.is_built_with_cuda()
tf.config.list_physical_devices("GPU")

# Top-k metrics
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = "top3_accuracy"
top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
top5_acc.__name__ = "top5_accuracy"


def train(model_path, restore, epochs, model, optim, datagen, testgen, data,
          cb_list, batch_size, verbosity=2):
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]

    if path.exists(model_path):
        if restore:
            model = load_model(model_path,
                               custom_objects={
                                   "top3_accuracy": top3_acc,
                                   "top5_accuracy": top5_acc
                               })
        else:
            remove(model_path)

    model.compile(optim,
                  loss="categorical_crossentropy",
                  metrics=["accuracy", top3_acc, top5_acc])

    history = model.fit_generator(datagen.flow(X_train,
                                               y_train,
                                               batch_size=batch_size),
                                  steps_per_epoch=len(X_train) / batch_size,
                                  epochs=epochs,
                                  verbose=verbosity,
                                  callbacks=cb_list,
                                  validation_data=testgen.flow(
                                      X_val, y_val, batch_size=batch_size),
                                  validation_steps=len(X_val) / batch_size)

    model.save(model_path)

    return history


def show_history(history):
    history = history.history

    # Plot training & validation loss values
    plot_history(history["loss"], history["val_loss"], "Categorical Cross-Entropy Loss")

    # Plot training & validation top-1 accuracy values
    plot_history(history["accuracy"], history["val_accuracy"],
                 "Top-1 Accuracy")

    # Plot training & validation top-3 accuracy values
    plot_history(history["top3_accuracy"], history["val_top3_accuracy"],
                 "Top-3 Accuracy")

    # Plot training & validation top-5 accuracy values
    plot_history(history["top5_accuracy"], history["val_top5_accuracy"],
                 "Top-5 Accuracy")


def plot_history(train, val, metric):
    plt.plot(train)
    plt.plot(val)
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.show()


def shuffle_channels(img):
    rand = random.randint(1, 10)
    if rand <= 8:
        return img
    img = np.moveaxis(img, -1, 0)
    np.random.shuffle(img)
    return np.moveaxis(img, 0, -1)
