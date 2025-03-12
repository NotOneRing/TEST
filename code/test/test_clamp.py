import numpy as np
import torch
import tensorflow as tf
import unittest

from util.torch_to_tf import torch_clamp

class TestClamp(unittest.TestCase):
    def test_clamp(self):
        # Create random input data
        np_input = np.random.randn(5, 5) * 10  # 5x5 array with values from a normal distribution

        # Convert to PyTorch and TensorFlow tensors
        torch_input = torch.tensor(np_input)
        tf_input = tf.convert_to_tensor(np_input, dtype=tf.float32)

        # Apply clamp operations
        torch_clipped = torch.clamp(torch_input, min=-5, max=5)
        torch_clip_clipped = torch_clamp(tf_input, min=-5, max=5)

        # Convert results to NumPy for comparison
        torch_clipped_np = torch_clipped.numpy()
        torch_clip_clipped_np = torch_clip_clipped.numpy()

        # # Print results for debugging
        # print("Torch clipped result:\n", torch_clipped_np)
        # print("Torch_clip (TensorFlow) clipped result:\n", torch_clip_clipped_np)
        # print("Are the results identical? ", np.allclose(torch_clipped_np, torch_clip_clipped_np))

        # Assert that the results are identical
        self.assertTrue(np.allclose(torch_clipped_np, torch_clip_clipped_np),
                        "PyTorch and TensorFlow clamp operations should produce identical results")

if __name__ == "__main__":
    unittest.main()
