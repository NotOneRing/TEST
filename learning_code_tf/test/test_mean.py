from util.torch_to_tf import torch_mean
import tensorflow as tf


import numpy as np

a = tf.convert_to_tensor(np.array([[1,2],[3,4]]))

b = torch_mean(a, dim=1)

print("b = ", b)

b = tf.reduce_mean(a)

print("b = ", b)
