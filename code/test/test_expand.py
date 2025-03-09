
import torch

import tensorflow as tf

from util.torch_to_tf import torch_tensor_expand

import numpy as np

def test_expand():
    # create a 1D Tensor
    tensor = torch.tensor([1, 2, 3])

    # expand to a 3x3 matrix
    expanded_tensor_1 = tensor.expand(3, 3)

    print(expanded_tensor_1)

    expanded_tensor_2 = tensor.expand([3, 3])

    print(expanded_tensor_2)


    # create a 1D Tensor
    tensor = tf.constant([1, 2, 3])


    # expand to a 3x3 matrix
    expanded_tensor_tf1 = torch_tensor_expand(tensor, [3, 3])

    print(expanded_tensor_tf1)


    expanded_tensor_tf2 = torch_tensor_expand(tensor, 3, 3)

    print(expanded_tensor_tf2)

    assert np.allclose(expanded_tensor_1.numpy(), expanded_tensor_tf1.numpy())
    assert np.allclose(expanded_tensor_2.numpy(), expanded_tensor_tf2.numpy())


test_expand()






