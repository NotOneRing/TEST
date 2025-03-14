import unittest
import torch
import numpy as np
import tensorflow as tf
from util.torch_to_tf import torch_tensor_view


class TestView(unittest.TestCase):
    """Test class for testing the torch_tensor_view function against PyTorch's view method."""

    def setUp(self):
        """Set up test fixtures, creating the test tensors."""
        # Create a PyTorch tensor for testing
        self.torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Create a TensorFlow tensor with the same values
        np_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.tf_tensor = tf.convert_to_tensor(np_array)

    def test_view_same_shape(self):
        """Test view operation with the same shape (3, 3)."""
        # PyTorch view operation
        torch_result = self.torch_tensor.view(3, 3)
        
        # TensorFlow equivalent using torch_tensor_view
        tf_result = torch_tensor_view(self.tf_tensor, 3, 3)
        
        # Verify the results are the same
        self.assertTrue(np.allclose(torch_result.numpy(), tf_result.numpy()))

    def test_view_add_dimension(self):
        """Test view operation adding a dimension (3, 3, 1)."""
        # PyTorch view operation with tuple arguments
        torch_result = self.torch_tensor.view(3, 3, 1)
        
        # TensorFlow equivalent using torch_tensor_view
        tf_result = torch_tensor_view(self.tf_tensor, 3, 3, 1)
        
        # Verify the results are the same
        self.assertTrue(np.allclose(torch_result.numpy(), tf_result.numpy()))

    def test_view_with_list(self):
        """Test view operation with a list argument [3, 3, 1]."""
        # PyTorch view operation with list argument
        torch_result = self.torch_tensor.view([3, 3, 1])
        
        # TensorFlow equivalent using torch_tensor_view
        tf_result = torch_tensor_view(self.tf_tensor, [3, 3, 1])
        
        # Verify the results are the same
        self.assertTrue(np.allclose(torch_result.numpy(), tf_result.numpy()))


if __name__ == '__main__':
    unittest.main()
