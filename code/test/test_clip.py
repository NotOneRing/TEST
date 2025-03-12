import unittest
import numpy as np
import torch
import tensorflow as tf

from util.torch_to_tf import torch_clip


class TestClip(unittest.TestCase):
    def test_clip(self):
        """Test that torch_clip function produces the same results as torch.clip."""
        # Create random input data
        np_input = np.random.randn(5, 5) * 10  # 5x5 array with values from a normal distribution

        # Convert to torch and tensorflow tensors
        torch_input = torch.tensor(np_input)
        tf_input = tf.convert_to_tensor(np_input, dtype=tf.float32)

        # Apply clipping operations
        torch_clipped = torch.clip(torch_input, min=-5, max=5)
        torch_clip_clipped = torch_clip(tf_input, min=-5, max=5)

        # Convert results to numpy for comparison
        torch_clipped_np = torch_clipped.numpy()
        torch_clip_clipped_np = torch_clip_clipped.numpy()

        # # Print results for debugging
        # print("Torch clipped result:\n", torch_clipped_np)
        # print("Torch_clip (TensorFlow) clipped result:\n", torch_clip_clipped_np)
        # print("Are the results identical? ", np.allclose(torch_clipped_np, torch_clip_clipped_np))

        # Assert that the results are identical
        self.assertTrue(np.allclose(torch_clipped_np, torch_clip_clipped_np))


if __name__ == '__main__':
    unittest.main()
