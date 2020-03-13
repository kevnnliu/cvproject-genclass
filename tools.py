import numpy as np
import cv2
import os
import random

def load_data(dataset='train'):
    print('Loading ' + dataset + ' data\n')
    image_folder = 'data/tiny-imagenet-200/'
    X = None
    y = None
    if dataset == 'train':
        X = np.load(image_folder + 'X_train.npy')
        y = np.load(image_folder + 'y_train.npy')
    elif dataset == 'val':
        X = np.load(image_folder + 'X_val.npy')
        y = np.load(image_folder + 'y_val.npy')
    elif dataset == 'test':
        X = np.load(image_folder + 'X_test.npy')
    else:
        raise ValueError("Dataset must be either 'train', 'val' or 'test'\n")
    
    print('Finished loading data\n')

    return X, y

def normalize_image(image):
    ROWS = 64
    COLUMNS = 64
    cv_image = cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (ROWS, COLUMNS), interpolation=cv2.INTER_CUBIC)
    return cv_image / 255.

def prepare_data():
    print('Preparing data\n')
    image_folder = 'data/tiny-imagenet-200/'
    class_labels = {}
    index = 0
    print('Loading labels\n')
    with open(image_folder + 'wnids.txt', 'r') as labels_file:
        for line in labels_file:
            label = line.strip()
            class_labels[label] = index
            index += 1
    print('Done\n')

    train_folder = image_folder + 'train/'
    val_folder = image_folder + 'val/'
    test_folder = image_folder + 'test/'

    train_images = []

    print('Loading training images\n')
    for label in class_labels.keys():
        class_folder_train = train_folder + label + '/images/'
        train_images.extend([class_folder_train + '{}'.format(i) for i in os.listdir(class_folder_train)])
    print('Done\n')

    print('Loading validation images\n')
    val_images = [val_folder + 'images/{}'.format(i) for i in os.listdir(val_folder + 'images/')]
    print('Done\n')

    print('Loading test images\n')
    test_images = [test_folder + 'images/{}'.format(i) for i in os.listdir(test_folder + 'images/')]
    print('Done\n')

    print('Shuffling\n')
    random.seed(91387264)
    random.shuffle(train_images)
    random.shuffle(val_images)
    random.shuffle(test_images)
    print('Done\n')

    X_train = []
    y_train = []

    X_val = []
    y_val =[]

    X_test = []
    # No need for y_test

    print('Formatting training images\n')
    for image in train_images:
        X_train.append(normalize_image(image))
        label = image[29 : 38]
        encoding = np.zeros(200)
        encoding[class_labels[label]] = 1
        y_train.append(encoding)
    print('Done\n')

    print('Getting validation labels\n')
    val_labels = {}
    with open(val_folder + 'val_annotations.txt', 'r') as val_labels_file:
        for line in val_labels_file:
            line = line.strip().split()
            val_labels[line[0]] = line[1]
    print('Done\n')

    print('Formatting validation images\n')
    for image in val_images:
        X_val.append(normalize_image(image))
        actual_image = image[34 :]
        label = val_labels[actual_image]
        encoding = np.zeros(200)
        encoding[class_labels[label]] = 1
        y_val.append(encoding)
    print('Done\n')

    print('Formatting test images\n')
    for image in test_images:
        X_test.append(normalize_image(image))
    print('Done\n')

    print('Converting to numpy arrays\n')
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    X_test = np.array(X_test)
    # No need for y_test
    print('Done\n')

    print('Saving numpy arrays\n')
    np.save(image_folder + 'X_train', X_train)
    np.save(image_folder + 'y_train', y_train)

    np.save(image_folder + 'X_val', X_val)
    np.save(image_folder + 'y_val', y_val)

    np.save(image_folder + 'X_test', X_test)
    print('Done\n')

    print('Finished preparing data\n')
