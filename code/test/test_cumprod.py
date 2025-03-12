import unittest
import torch
import tensorflow as tf
import numpy as np
from util.torch_to_tf import torch_cumprod


class TestCumprod(unittest.TestCase):
    def test_1d_tensor(self):
        """Test cumulative product on 1D tensors in both PyTorch and TensorFlow."""
        # PyTorch implementation
        torch_tensor = torch.tensor([1, 2, 3, 4])
        torch_result = torch.cumprod(torch_tensor, dim=0)
        
        # TensorFlow implementation
        tf_tensor = tf.constant([1, 2, 3, 4])
        tf_result = torch_cumprod(tf_tensor, dim=0)
        
        # Assert results are equal
        self.assertTrue(np.array_equal(torch_result.numpy(), tf_result.numpy()))
    
    def test_2d_tensor_dim0(self):
        """Test cumulative product on 2D tensors along dimension 0."""
        # PyTorch implementation
        torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        torch_result = torch.cumprod(torch_tensor, dim=0)
        
        # TensorFlow implementation
        tf_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
        tf_result = torch_cumprod(tf_tensor, dim=0)
        
        # Assert results are equal
        self.assertTrue(np.array_equal(torch_result.numpy(), tf_result.numpy()))
        
        # Verify expected values
        expected = torch.tensor([[1, 2, 3], [4, 10, 18]])
        self.assertTrue(torch.all(torch.eq(torch_result, expected)))
    
    def test_2d_tensor_dim1(self):
        """Test cumulative product on 2D tensors along dimension 1."""
        # PyTorch implementation
        torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        torch_result = torch.cumprod(torch_tensor, dim=1)
        
        # TensorFlow implementation
        tf_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
        tf_result = torch_cumprod(tf_tensor, dim=1)
        
        # Assert results are equal
        self.assertTrue(np.array_equal(torch_result.numpy(), tf_result.numpy()))
        
        # Verify expected values
        expected = torch.tensor([[1, 2, 6], [4, 20, 120]])
        self.assertTrue(torch.all(torch.eq(torch_result, expected)))


if __name__ == '__main__':
    unittest.main()


