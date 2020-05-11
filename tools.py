import numpy as np
import cv2
import os
import random

DEFAULT_IMAGE_FOLDER = "data/tiny-imagenet-200/"

def get_word_labels(img_folder=DEFAULT_IMAGE_FOLDER):
    image_folder = img_folder
    class_words = {}
    print("Loading words\n")
    with open(image_folder + "words.txt", "r") as labels_file:
        for line in labels_file:
            label = line.strip().split()
            class_words[label[0]] = label[1 :]
    print("Done\n")

    return class_words

def get_label_dict(img_folder=DEFAULT_IMAGE_FOLDER):
    image_folder = img_folder
    class_labels = {}
    index = 0
    print("Loading labels\n")
    with open(image_folder + "wnids.txt", "r") as labels_file:
        for line in labels_file:
            label = line.strip()
            class_labels[index] = label
            index += 1
    print("Done\n")

    return class_labels

def get_box_dict_val(img_folder=DEFAULT_IMAGE_FOLDER):
    image_folder = img_folder+"val/"
    class_boxes = {}
    print("Loading bounding boxes\n")
    with open(image_folder + "val_annotations.txt", "r") as labels_file:
        for line in labels_file:
            label = line.strip().split()
            img_name = label[0]
            n_label = label[1]
            box_coords = label[2:]
            class_boxes[img_name] = box_coords
    print("Done\n")

    return class_boxes


def load_data(dataset="train", img_folder=DEFAULT_IMAGE_FOLDER):
    print("Loading " + dataset + " data\n")
    image_folder = img_folder
    X = None
    y = None
    if dataset == "train":
        X = np.load(image_folder + "X_train.npy")
        y = np.load(image_folder + "y_train.npy")
    elif dataset == "val":
        X = np.load(image_folder + "X_val.npy")
        y = np.load(image_folder + "y_val.npy")
    elif dataset == "test":
        X = np.load(image_folder + "X_test.npy")
    else:
        raise ValueError("Dataset must be either 'train', 'val' or 'test'\n")
    
    print("Finished loading " + dataset + " data\n")

    return X, y

def prepare_data(img_folder=DEFAULT_IMAGE_FOLDER):
    print("Preparing data\n")
    image_folder = img_folder
    class_labels = {}
    index = 0
    print("Loading labels\n")
    with open(image_folder + "wnids.txt", "r") as labels_file:
        for line in labels_file:
            label = line.strip()
            class_labels[label] = index
            index += 1
    print("Done\n")

    train_folder = image_folder + "train/"
    val_folder = image_folder + "val/"
    test_folder = image_folder + "test/"

    train_images = []

    print("Loading training images\n")
    for label in class_labels.keys():
        class_folder_train = train_folder + label + "/images/"
        train_images.extend([class_folder_train + "{}".format(i) for i in os.listdir(class_folder_train)])
    print("Done\n")

    print("Loading validation images\n")
    val_images = [val_folder + "images/{}".format(i) for i in os.listdir(val_folder + "images/")]
    print("Done\n")

    print("Loading test images\n")
    test_images = [test_folder + "images/{}".format(i) for i in os.listdir(test_folder + "images/")]
    print("Done\n")

    print("Shuffling\n")
    random.seed(91387264)
    random.shuffle(train_images)
    random.shuffle(val_images)
    # Don"t shuffle test images
    print("Done\n")

    X_train = []
    y_train = []

    X_val = []
    y_val =[]

    X_test = []
    # No need for y_test

    print("Formatting training images\n")
    for image in train_images:
        cv_image = read_resize(image)
        X_train.append(cv_image)
        label = image[29 : 38]
        encoding = class_labels[label]
        y_train.append(encoding)
    print("Done\n")

    print("Getting validation labels\n")
    val_labels = {}
    with open(val_folder + "val_annotations.txt", "r") as val_labels_file:
        for line in val_labels_file:
            line = line.strip().split()
            val_labels[line[0]] = line[1]
    print("Done\n")

    print("Formatting validation images\n")
    for image in val_images:
        cv_image = read_resize(image)
        X_val.append(cv_image)
        actual_image = image[34 :]
        label = val_labels[actual_image]
        encoding = class_labels[label]
        y_val.append(encoding)
    print("Done\n")

    print("Formatting test images\n")
    for image in test_images:
        cv_image = read_resize(image)
        X_test.append(cv_image)
    print("Done\n")

    print("Converting to numpy arrays\n")
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    X_test = np.array(X_test)
    # No need for y_test
    print("Done\n")

    print("Saving numpy arrays\n")
    np.save(image_folder + "X_train", X_train)
    np.save(image_folder + "y_train", y_train)

    np.save(image_folder + "X_val", X_val)
    np.save(image_folder + "y_val", y_val)

    np.save(image_folder + "X_test", X_test)
    print("Done\n")

    print("Finished preparing data\n")

def read_resize(img, factor=1):
    cv_image = cv2.imread(img, cv2.IMREAD_COLOR)
    dim = (int)(64 * factor)
    return cv2.resize(cv_image, (dim, dim), interpolation=cv2.INTER_LANCZOS4) / 255.
