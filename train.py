import tensorflow as tf
from keras import backend as kb

import pathlib
import numpy as np
import cv2
import pandas
import matplotlib as plt
import os
import random

# Use these to test check for gpu availability
# tf.test.is_built_with_cuda()
# tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

class_labels = []
with open('data/tiny-imagenet-200/wnids.txt', 'r') as labels_file:
    line = labels_file.readline()
    while line:
        class_labels.append(line.strip())
        line = labels_file.readline()

for c in class_labels:
    print(c)
