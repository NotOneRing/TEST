

from util.torch_to_tf import torch_arange


import numpy as np

def test_arange():
    tensor_tf1 = torch_arange(0, 10)
    print(tensor_tf1)
    # output：tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # create tensor from 1 to 9 with stride length of 2
    tensor_tf2 = torch_arange(1, 10, step=2)
    print(tensor_tf2)

    print("type(tensor_tf2) = ", type(tensor_tf2))


    import torch

    tensor1 = torch.arange(0, 10)
    print(tensor1)
    # Output: tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # create tensor from 1 to 9 with stride length of 2
    tensor2 = torch.arange(1, 10, step=2)
    print(tensor2)

    assert np.allclose(tensor_tf1.numpy(), tensor1.numpy())

    assert np.allclose(tensor_tf2.numpy(), tensor2.numpy())




test_arange()












