import unittest
import numpy as np
import torch
import tensorflow as tf

from util.torch_to_tf import torch_tensor_clamp_

class TestClamp(unittest.TestCase):
    def test_clamp_(self):
        # Create random input data
        np_input = np.random.randn(5, 5) * 10  # 5x5 array with values from a normal distribution

        # Convert to torch tensor
        torch_input = torch.tensor(np_input)
        
        # Convert to tensorflow tensor
        tf_input = tf.convert_to_tensor(np_input, dtype=tf.float32)

        # Apply torch's clamp_ function
        torch.clamp_(torch_input, min=-5, max=5)

        # Create a TensorFlow variable from the input
        variable = tf.Variable(tf_input)

        # Apply the custom torch_tensor_clamp_ function
        torch_tensor_clamp_(variable, min=-5.0, max=5.0)
        tf_input = tf.convert_to_tensor(variable)

        # Convert results to numpy for comparison
        torch_clipped_np = torch_input.numpy()
        torch_clip_clipped_np = tf_input.numpy()

        # print("Torch clipped result:\n", torch_clipped_np)
        # print("Torch_clip (TensorFlow) clipped result:\n", torch_clip_clipped_np)
        # print("Are the results identical? ", np.allclose(torch_clipped_np, torch_clip_clipped_np))

        # Assert that the results are identical
        self.assertTrue(np.allclose(torch_clipped_np, torch_clip_clipped_np))

if __name__ == '__main__':
    unittest.main()
