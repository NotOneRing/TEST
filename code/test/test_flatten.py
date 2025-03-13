import tensorflow as tf
import torch
import unittest
import numpy as np

from util.torch_to_tf import torch_flatten


class TestFlatten(unittest.TestCase):
    def test_flatten(self):
        # Step 1: Create a NumPy array for feats
        batch_size = 10
        height = 6
        width = 6
        channels = 128

        feats = np.random.rand(batch_size, height, width, channels).astype(np.float32)

        # Step 2: Convert to TensorFlow tensor
        feats_tf = tf.convert_to_tensor(feats)

        # Step 3: Convert to PyTorch tensor
        feats_torch = torch.tensor(feats)

        # Step 4: Flatten the second and third dimensions (height and width) for each format
        # NumPy
        feats_np_flatten = feats.reshape(feats.shape[0], -1, feats.shape[-1])

        # TensorFlow
        feats_tf_flatten = tf.reshape(feats_tf, [feats_tf.shape[0], -1, feats_tf.shape[-1]])

        output = torch_flatten(feats_tf, 1, 2)

        print("output = ", output)

        # PyTorch
        feats_torch_flatten = feats_torch.flatten(start_dim=1, end_dim=2)

        # Step 5: Compare the results
        # Convert TensorFlow and PyTorch tensors back to NumPy arrays for comparison
        feats_tf_flatten_np = feats_tf_flatten.numpy()
        feats_torch_flatten_np = feats_torch_flatten.numpy()
        output_np = output.numpy()

        # Assert that all implementations produce the same result
        self.assertTrue(np.allclose(feats_np_flatten, feats_tf_flatten_np), 
                        "NumPy and TensorFlow implementations don't match")
        self.assertTrue(np.allclose(feats_np_flatten, feats_torch_flatten_np), 
                        "NumPy and PyTorch implementations don't match")
        self.assertTrue(np.allclose(feats_tf_flatten_np, feats_torch_flatten_np), 
                        "TensorFlow and PyTorch implementations don't match")
        self.assertTrue(np.allclose(feats_np_flatten, output_np), 
                        "NumPy and custom torch_flatten implementations don't match")

        # Check shapes are consistent
        self.assertEqual(feats_np_flatten.shape, feats_tf_flatten.shape, 
                        "NumPy and TensorFlow shapes don't match")
        self.assertEqual(feats_np_flatten.shape, feats_torch_flatten.shape, 
                        "NumPy and PyTorch shapes don't match")
        self.assertEqual(feats_np_flatten.shape, output.shape, 
                        "NumPy and custom torch_flatten shapes don't match")


if __name__ == "__main__":
    unittest.main()











