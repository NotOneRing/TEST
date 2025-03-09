import tensorflow as tf
import torch

import numpy as np

def test_squeeze():

    # create a tensor, containing dimension of 1
    x_torch = torch.tensor([[[1, 2, 3], [4, 5, 6]]])  # shape is (1, 2, 3)

    # use torch.squeeze to remove dimension with value 1
    x_torch_squeezed = torch.squeeze(x_torch)
    print("PyTorch Squeeze Output:")
    print(x_torch_squeezed)
    print("Shape after squeeze:", x_torch_squeezed.shape)  # output (2, 3)



    # create a tensor, containing dimension of 1
    x_tf = tf.constant([[[1, 2, 3], [4, 5, 6]]])  # shape is (1, 2, 3)

    # use tf.squeeze to remove dimension with value 1
    x_tf_squeezed = tf.squeeze(x_tf)
    print("TensorFlow Squeeze Output:")
    print(x_tf_squeezed)
    print("Shape after squeeze:", x_tf_squeezed.shape)  # output (2, 3)


    assert np.allclose(x_torch_squeezed.numpy(), x_tf_squeezed.numpy())

    # create a tensor, containing dimension of 1
    x_torch = torch.tensor([[[1, 2, 3], [4, 5, 6]]])  # shape is (1, 2, 3)

    # use tf.squeeze to remove dimension 0 (the zeroth dimension) with value 1
    x_torch_squeezed = torch.squeeze(x_torch, dim=0)
    print("PyTorch Squeeze Output (dim=0):")
    print(x_torch_squeezed)
    print("Shape after squeeze:", x_torch_squeezed.shape)  # output (2, 3)



    # create a tensor, containing dimension of 1
    x_tf = tf.constant([[[1, 2, 3], [4, 5, 6]]])  # shape is (1, 2, 3)

    # use tf.squeeze to remove dimension 0 (the zeroth dimension) with value 1
    x_tf_squeezed = tf.squeeze(x_tf, axis=0)
    print("TensorFlow Squeeze Output (axis=0):")
    print(x_tf_squeezed)
    print("Shape after squeeze:", x_tf_squeezed.shape)  # shape is (2, 3)

    assert np.allclose(x_torch_squeezed.numpy(), x_tf_squeezed.numpy())


test_squeeze()


