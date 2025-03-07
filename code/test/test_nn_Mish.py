import torch
import torch.nn as nn
import numpy as np



def test_Mish():
    # create a Mish activation
    mish = nn.Mish()

    # input tensor
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])

    # apply mish activation
    output = mish(input_tensor)
    print(output)



    import tensorflow as tf
    from util.torch_to_tf import nn_Mish

    # create a layer of Mish activation
    mish = nn_Mish()

    # input tensor
    input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

    # apply mish activation
    output_tf = mish(input_tensor_tf)
    print(output_tf)

    assert np.allclose(output, output_tf)



test_Mish()







