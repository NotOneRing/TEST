import torch
import torch.nn as nn

import numpy as np

def test_ReLU():
    # create a layer of ReLU
    relu = nn.ReLU()

    # input tensor
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])

    # apply relu activation
    output = relu(input_tensor)
    print(output)



    import tensorflow as tf
    from util.torch_to_tf import nn_ReLU

    # create a layer of ReLU
    relu = nn_ReLU()

    # input tensor
    input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

    # apply relu activation
    output_tf = relu(input_tensor_tf)
    print(output_tf)

    assert np.allclose(output, output_tf)



test_ReLU()







