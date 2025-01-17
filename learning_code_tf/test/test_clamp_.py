import numpy as np
import torch
import tensorflow as tf

from util.torch_to_tf import torch_tensor_clamp_

def test_clamp_():

    np_input = np.random.randn(5, 5) * 10  # 5x5 array with values from a normal distribution

    torch_input = torch.tensor(np_input)
    # tf_input = tf.Variable(np_input, dtype=tf.float32)
    tf_input = tf.convert_to_tensor(np_input, dtype=tf.float32)
    # tf_input = tf.Variable(np_input, dtype=tf.float32)



    torch.clamp_(torch_input, min=-5, max=5)

    print("tf_input = ", tf_input)

    variable = tf.Variable(tf_input)

    # print("variable = ", variable)

    torch_tensor_clamp_(variable, min=-5.0, max=5.0)
    tf_input = tf.convert_to_tensor(variable)

    torch_clipped_np = torch_input.numpy()
    torch_clip_clipped_np = tf_input.numpy()

    print("Torch clipped result:\n", torch_clipped_np)
    print("Torch_clip (TensorFlow) clipped result:\n", torch_clip_clipped_np)
    print("Are the results identical? ", np.allclose(torch_clipped_np, torch_clip_clipped_np))

    assert np.allclose(torch_clipped_np, torch_clip_clipped_np)



test_clamp_()





