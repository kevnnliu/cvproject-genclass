import tensorflow as tf
from keras.models import load_model
import models

from os import path, remove
import matplotlib.pyplot as plt

# Use these to test check for gpu availability
tf.test.is_built_with_cuda()
tf.config.list_physical_devices("GPU")

def train(model_path, restore, epochs, model, optim, datagen, data, cb_list, batch_size=32):
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]

    if path.exists(model_path):
        if restore:
            model = load_model(model_path)
        else:
            remove(model_path)

    model.summary()
    model.compile(optim, loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), 
                                    steps_per_epoch=len(X_train) / batch_size, 
                                    epochs=epochs, verbose=1, callbacks=cb_list, 
                                    validation_data=datagen.flow(X_val, y_val, batch_size=batch_size),
                                    validation_steps=len(X_val) / batch_size)

    model.save(model_path)

    return history

def show_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.show()

def shuffle_channels(img):
    img_transposed = tf.transpose(img, [2, 0, 1])
    shuffled = tf.random.shuffle(img_transposed)
    return tf.transpose(shuffled, [1, 2, 0])
