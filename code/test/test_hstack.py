import torch

import tensorflow as tf

from util.torch_to_tf import torch_hstack, torch_tensor_view

import numpy as np


def test_hstack():
    a = torch.tensor([[1, 2], [3, 4]]).reshape(1, 1, 4)
    b = torch.tensor([[5, 6], [7, 8]]).reshape(1, 1, 4)

    result = torch.hstack((a, b))
    print(result)

    print(result.shape)

    # output:
    # tensor([[1, 2, 5, 6],
    #         [3, 4, 7, 8]])

    a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

    a = torch_tensor_view(a, 1, 1, 4)
    b = torch_tensor_view(b, 1, 1, 4)

    # print("a = ", a)
    # print("b = ", b)

    result_tf = torch_hstack((a, b))

    print("result_tf = ", result_tf)

    print(result_tf)
    assert np.allclose(result.numpy(), result_tf.numpy())
    # output:
    # tensor([[1, 2, 5, 6],
    #         [3, 4, 7, 8]])



test_hstack()






