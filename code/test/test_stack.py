import torch

import tensorflow as tf

import numpy as np

from util.torch_to_tf import torch_stack

def test_stack():
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    b = tf.constant([[7, 8, 9], [10, 11, 12]])

    torch_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    torch_b = torch.tensor([[7, 8, 9], [10, 11, 12]])


    s1 = torch_stack([a, b], dim=0)

    torch_s1 = torch.stack([torch_a, torch_b], dim=0)

    print("s1 (stack axis=0):\n", s1.numpy())

    print("torch_s1 (stack axis=0):\n", torch_s1.numpy())

    assert np.allclose( s1.numpy(),  torch_s1.numpy() )



    s2 = torch_stack([a, b], dim=1)

    torch_s2 = torch.stack([torch_a, torch_b], dim=1)

    print("\n")

    print("s2 (stack axis=1):\n", s2.numpy())

    print("torch_s2 (stack axis=1):\n", torch_s2.numpy())

    assert np.allclose( s2.numpy(),  torch_s2.numpy() )





    torch_s3 = torch.stack([torch_a, torch_b])


    print("s3 (stack axis=None):\n", torch_s3.numpy())

    s3 = torch_stack([a, b])

    print("s3 (stack axis=None):\n", s3.numpy())

    print("s3.shape (stack axis=None):\n", s3.numpy().shape)


    assert np.allclose( s3.numpy(),  torch_s3.numpy() )


test_stack()


































