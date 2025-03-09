
import torch

import numpy as np

import tensorflow as tf

from util.torch_to_tf import torch_argmax

def test_argmax():
    # example tensor
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # get the max index along the zeroth dimension
    result_torch1 = torch.argmax(tensor, dim=0)
    print(result_torch1)
    # Output: tensor([1, 1, 1])  # get the max index along the 0 dimension

    # get the max index along the first dimension
    result_torch2 = torch.argmax(tensor, dim=1)
    print(result_torch2)
    # Output: tensor([2, 2])  # get the max index along the first dimension

    # example tensor
    tensor = np.array([[1, 2, 3], [4, 5, 6]])

    tensor = tf.convert_to_tensor(tensor)

    # get the max index along the zeroth dimension
    result_tf1 = torch_argmax(tensor, dim=0)
    print(result_tf1)
    # Output: tensor([1, 1, 1])  # get the max index along the zeroth dimension

    # get the max index along the first dimension
    result_tf2 = torch_argmax(tensor, dim=1)
    print(result_tf2)
    # Output: tensor([2, 2])  # get the max index along the first dimension


    assert np.allclose(result_torch1.numpy(), result_tf1.numpy())
    assert np.allclose(result_torch2.numpy(), result_tf2.numpy())
    


test_argmax()







