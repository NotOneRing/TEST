
import tensorflow as tf

import torch

import numpy as np

def test_squeeze():

    x = np.array(range(27)).reshape(3, 3, 3)

    # b = torch_mean(a, dim=1)

    # print("b = ", b)

    # b = tf.reduce_mean(a)

    # print("b = ", b)

    self_event_ndims = 1

    x_torch = torch.tensor(x)

    x_expands = x_torch.unsqueeze(self_event_ndims)

    print("x_expands = ", x_expands)

    print("x_expands.shape = ", x_expands.shape)

    x_tf = tf.convert_to_tensor(x)

    x_expands_tf = tf.expand_dims(x_tf, axis=self_event_ndims)

    print("x_expands_tf = ", x_expands_tf)

    assert np.allclose(x_expands_tf.numpy(), x_expands.numpy())



test_squeeze()













