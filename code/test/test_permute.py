import unittest
from util.torch_to_tf import torch_tensor_permute

import torch
import tensorflow as tf
import numpy as np

class TestPermute(unittest.TestCase):
    """Test case for the permute operation between PyTorch and TensorFlow."""
    
    def test_comparison(self):
        """
        Test that torch.permute and torch_tensor_permute produce the same results.
        Compares the output of PyTorch's permute operation with our custom
        torch_tensor_permute function for TensorFlow.
        """
        # Create a random PyTorch tensor
        input_tensor_torch = torch.randn(1, 2, 3)
        
        # Apply permute operation in PyTorch
        output_tensor_torch = input_tensor_torch.permute(1, 2, 0)
        
        # print(output_tensor_torch.shape)
        
        # Convert PyTorch tensor to TensorFlow tensor
        input_tensor_tf = tf.convert_to_tensor(input_tensor_torch.numpy())
        
        # Apply permute operation in TensorFlow
        # output_tensor_tf = tf.transpose(input_tensor_tf, perm=[1, 2, 0])
        # output_tensor_tf = torch_tensor_permute(input_tensor_tf, 1, 2, 0)
        output_tensor_tf = torch_tensor_permute(input_tensor_tf, (1, 2, 0))
        
        # print(output_tensor_tf.shape)

        # print("output_tensor_torch = ", output_tensor_torch)
        # print("output_tensor_tf = ", output_tensor_tf)

        # Assert that the results are the same
        self.assertTrue(np.allclose(output_tensor_torch.numpy(), output_tensor_tf.numpy()))


if __name__ == '__main__':
    unittest.main()
