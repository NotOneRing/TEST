import torch
import torch.nn as nn

import numpy as np

def test_Tanh():
    # create a layer of Tanh
    tanh = nn.Tanh()

    # input tensor
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])

    # create tanh activation
    output = tanh(input_tensor)
    print(output)



    import tensorflow as tf
    from util.torch_to_tf import nn_Tanh

    # creata a layer of Tanh
    tanh = nn_Tanh()

    # input tensor
    input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

    # apply tanh activation
    output_tf = tanh(input_tensor_tf)
    print(output_tf)

    assert np.allclose(output, output_tf)



test_Tanh()







