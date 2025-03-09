import torch

import numpy as np

import tensorflow as tf

from util.torch_to_tf import torch_sqrt

def test_sqrt():

    sigma = torch.tensor([1.0, 2.0, 3.0])
    var = torch.sqrt(sigma)
    print(var)  
    
    sigma = tf.constant([1.0, 2.0, 3.0])
    var_tf = torch_sqrt(sigma)  # element-wise square
    print(var) 
    

    assert np.allclose(var.numpy(), var_tf.numpy())


test_sqrt()






























