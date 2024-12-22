import torch
import torch.nn.functional as F

import tensorflow as tf

from util.torch_to_tf import torch_randn


def test_nn_functional_pad():

    x = torch_randn(2,3,4)



    x_torch = torch.tensor(x.numpy())


    pad_value = 2

    x_torch_padded_torch = F.pad(x_torch, (pad_value, pad_value, pad_value, pad_value), mode='replicate')


    print("\nPyTorch nn.pad replicate padded tensor:")
    print(x_torch_padded_torch)


    from util.torch_to_tf import nn_functional_pad

    result = nn_functional_pad(x, (pad_value, pad_value, pad_value, pad_value), mode = "replicate")


    print("result.shape = ", result.shape)


    import numpy as np

    assert np.allclose(result.numpy(), x_torch_padded_torch.numpy())


    print("np.allclose(result.numpy(), x_torch_padded_torch.numpy()) = ", np.allclose(result.numpy(), x_torch_padded_torch.numpy()))

