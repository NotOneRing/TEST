import torch

import numpy as np

import tensorflow as tf

from util.torch_to_tf import torch_square

def test_square():

    sigma = torch.tensor([1.0, 2.0, 3.0])
    var = torch.square(sigma)  # element-wise square
    print(var)  # output: tensor([1., 4., 9.])

    sigma = tf.constant([1.0, 2.0, 3.0])
    var_tf = torch_square(sigma)  # element-wise square
    print(var)  # output: [1. 4. 9.]


    assert np.allclose(var.numpy(), var_tf.numpy())


test_square()






























