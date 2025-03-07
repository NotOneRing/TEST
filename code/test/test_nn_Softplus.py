import torch
import torch.nn as nn
import numpy as np



def test_Softplus():
    # create a layer of Softplus
    softplus = nn.Softplus()

    # input tensor
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])

    # apply softplus activation
    output = softplus(input_tensor)
    print(output)



    import tensorflow as tf
    from util.torch_to_tf import nn_Softplus

    # create a layer of Softplus
    softplus = nn_Softplus()

    # input tensor
    input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

    # apply softplus activation
    output_tf = softplus(input_tensor_tf)
    print(output_tf)

    assert np.allclose(output, output_tf)



test_Softplus()













