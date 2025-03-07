import torch
import torch.nn as nn
import numpy as np



def test_Identity():
    # createa a layer of Identity
    identity = nn.Identity()

    # input tensor
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])

    # apply identity activation
    output = identity(input_tensor)
    print(output)



    import tensorflow as tf
    from util.torch_to_tf import nn_Identity

    # creatae a layer of Identity
    identity = nn_Identity()

    # input tensor
    input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

    # apply Identity
    output_tf = identity(input_tensor_tf)
    print(output_tf)

    assert np.allclose(output, output_tf)



test_Identity()







