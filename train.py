import tensorflow as tf
from keras import backend as kb

# Use these to test check for gpu availability
tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
