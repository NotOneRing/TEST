import unittest
from util.torch_to_tf import torch_reshape

import torch
import tensorflow as tf
import numpy as np

# Original comment: def torch_reshape(input, shape):
#                   return tf.reshape(input, shape)

class TestReshape(unittest.TestCase):
    def test_reshape(self):
        """
        Test case for torch.reshape and torch_reshape
        Compares the results of PyTorch's reshape and our TensorFlow implementation
        """
        # Create a random tensor with a fixed shape
        input_tensor_torch = torch.randn(2, 3, 4)
        input_tensor_tf = tf.convert_to_tensor(input_tensor_torch.numpy())

        # Define target shape
        target_shape = (4, 6)

        # Perform reshape in PyTorch
        torch_result = torch.reshape(input_tensor_torch, target_shape)

        # Perform reshape in TensorFlow
        # tf_result = torch_reshape(input_tensor_tf, *target_shape)
        tf_result = torch_reshape(input_tensor_tf, target_shape)

        # Original comment: tf_result = torch_reshape(input_tensor_tf, target_shape)

        # Convert TensorFlow result to numpy for comparison
        tf_result_np = tf_result.numpy()

        # Check if the results match using unittest assertion
        self.assertTrue(np.allclose(torch_result.numpy(), tf_result_np),
                        "PyTorch and TensorFlow reshape results should match")
        
        # Additional assertions to verify the shape is correct
        self.assertEqual(torch_result.shape, target_shape,
                         "PyTorch result shape should match target shape")
        self.assertEqual(tf_result.shape, target_shape,
                         "TensorFlow result shape should match target shape")


if __name__ == '__main__':
    unittest.main()


