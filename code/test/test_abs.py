
import torch

sigma = torch.tensor([1.0, -2.0, 3.0])
var = torch.abs(sigma)
print(var)  # output: tensor([1., 4., 9.])


import tensorflow as tf

from util.torch_to_tf import torch_abs

sigma = tf.constant([1.0, -2.0, 3.0])
var = torch_abs(sigma)
print(var)  # output: [1. 4. 9.]


































