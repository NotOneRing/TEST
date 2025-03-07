import torch
import torch.nn as nn

import numpy as np

def test_GELU():
    # create a layer of GELU
    gelu = nn.GELU()

    # input tensor
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])

    # apply gelu activation
    output = gelu(input_tensor)
    print(output)



    import tensorflow as tf
    from util.torch_to_tf import nn_GELU

    # create a layer of GELU
    gelu = nn_GELU()

    # input tensor
    input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

    # apply gelu activation
    output_tf = gelu(input_tensor_tf)
    print(output_tf)

    assert np.allclose(output, output_tf)



test_GELU()













