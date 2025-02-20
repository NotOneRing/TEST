import numpy as np
import torch
import tensorflow as tf

from util.torch_to_tf import torch_clip

def test_clip():

    np_input = np.random.randn(5, 5) * 10  # 5x5 array with values from a normal distribution

    torch_input = torch.tensor(np_input)
    tf_input = tf.convert_to_tensor(np_input, dtype=tf.float32)

    # torch_clipped = torch.clamp(torch_input, min=-5, max=5)
    torch_clipped = torch.clip(torch_input, min=-5, max=5)
    torch_clip_clipped = torch_clip(tf_input, min=-5, max=5)

    torch_clipped_np = torch_clipped.numpy()
    torch_clip_clipped_np = torch_clip_clipped.numpy()

    print("Torch clipped result:\n", torch_clipped_np)
    print("Torch_clip (TensorFlow) clipped result:\n", torch_clip_clipped_np)
    print("Are the results identical? ", np.allclose(torch_clipped_np, torch_clip_clipped_np))

    assert np.allclose(torch_clipped_np, torch_clip_clipped_np)



test_clip()

