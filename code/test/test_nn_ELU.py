import torch
import torch.nn as nn

import numpy as np

def test_ELU():
    # create a layer of ELU activation
    elu = nn.ELU()

    # input tensor
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])

    # apply ELU activation
    output = elu(input_tensor)
    print(output)



    import tensorflow as tf
    from util.torch_to_tf import nn_ELU

    # create a layer of ELU activation
    elu = nn_ELU()

    # input tensor
    input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

    # apply ELU activation
    output_tf = elu(input_tensor_tf)
    print(output_tf)

    assert np.allclose(output, output_tf)



test_ELU()


