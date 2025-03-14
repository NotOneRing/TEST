import tensorflow as tf
import torch
import numpy as np
import unittest

from util.torch_to_tf import torch_squeeze

class TestSqueeze(unittest.TestCase):
    def test_squeeze_basic(self):
        """
        Test basic squeeze operation without specifying dimensions.
        Compares PyTorch and TensorFlow implementations.
        """
        # Create a tensor, containing dimension of 1
        x_torch = torch.tensor([[[1, 2, 3], [4, 5, 6]]])  # shape is (1, 2, 3)

        # Use torch.squeeze to remove dimension with value 1
        x_torch_squeezed = torch.squeeze(x_torch)
        # print("PyTorch Squeeze Output:")
        # print(x_torch_squeezed)
        # print("Shape after squeeze:", x_torch_squeezed.shape)  # output (2, 3)

        # Create a tensor, containing dimension of 1
        x_tf = tf.constant([[[1, 2, 3], [4, 5, 6]]])  # shape is (1, 2, 3)

        # Use tf.squeeze to remove dimension with value 1
        x_tf_squeezed = torch_squeeze(x_tf)
        # print("TensorFlow Squeeze Output:")
        # print(x_tf_squeezed)
        # print("Shape after squeeze:", x_tf_squeezed.shape)  # output (2, 3)

        # Verify that PyTorch and TensorFlow implementations produce the same result
        self.assertTrue(np.allclose(x_torch_squeezed.numpy(), x_tf_squeezed.numpy()))

    def test_squeeze_with_dimension(self):
        """
        Test squeeze operation with a specific dimension.
        Compares PyTorch and TensorFlow implementations.
        """
        # Create a tensor, containing dimension of 1
        x_torch = torch.tensor([[[1, 2, 3], [4, 5, 6]]])  # shape is (1, 2, 3)

        # Use torch.squeeze to remove dimension 0 (the zeroth dimension) with value 1
        x_torch_squeezed = torch.squeeze(x_torch, dim=0)
        # print("PyTorch Squeeze Output (dim=0):")
        # print(x_torch_squeezed)
        # print("Shape after squeeze:", x_torch_squeezed.shape)  # output (2, 3)

        # Create a tensor, containing dimension of 1
        x_tf = tf.constant([[[1, 2, 3], [4, 5, 6]]])  # shape is (1, 2, 3)

        # Use tf.squeeze to remove dimension 0 (the zeroth dimension) with value 1
        x_tf_squeezed = torch_squeeze(x_tf, dim=0)
        # print("TensorFlow Squeeze Output (axis=0):")
        # print(x_tf_squeezed)
        # print("Shape after squeeze:", x_tf_squeezed.shape)  # shape is (2, 3)

        # Verify that PyTorch and TensorFlow implementations produce the same result
        self.assertTrue(np.allclose(x_torch_squeezed.numpy(), x_tf_squeezed.numpy()))

if __name__ == '__main__':
    unittest.main()
