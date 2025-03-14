import unittest
import numpy as np
import tensorflow as tf
import torch

from util.torch_to_tf import torch_sum

class TestSum(unittest.TestCase):
    """Test class for verifying the torch_sum function implementation."""
    
    def test_sum_with_dim_and_keepdim(self):
        """Test torch_sum with dim=-1 and keepdim=True parameters.
        
        This test verifies that torch_sum behaves the same as torch.sum
        when using dimension and keepdim parameters.
        """
        # Create test array
        arr = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
        
        # Test with TensorFlow
        probs_tf = tf.constant(arr, dtype=tf.float32)
        self_probs_tf = probs_tf / torch_sum(probs_tf, dim=-1, keepdim=True)
        # print("self_probs_tf = ", self_probs_tf)
        
        # Test with PyTorch
        probs_torch = torch.tensor(arr)
        self_probs_torch = probs_torch / probs_torch.sum(-1, keepdim=True)
        # print("self_probs = ", self_probs_torch)
        
        # Verify results are the same
        self.assertTrue(np.allclose(self_probs_torch.numpy(), self_probs_tf.numpy()))
    
    def test_sum_without_parameters(self):
        """Test torch_sum without parameters.
        
        This test verifies that torch_sum behaves the same as torch.sum
        when using default parameters (summing all elements).
        """
        # Create test array
        arr = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
        
        # Test with TensorFlow
        probs_tf = tf.constant(arr, dtype=tf.float32)
        self_probs_tf = probs_tf / torch_sum(probs_tf)
        # print("self_probs_tf = ", self_probs_tf)
        
        # Test with PyTorch
        probs_torch = torch.tensor(arr)
        self_probs_torch = probs_torch / probs_torch.sum()
        # print("self_probs = ", self_probs_torch)
        
        # Verify results are the same
        self.assertTrue(np.allclose(self_probs_torch.numpy(), self_probs_tf.numpy()))

if __name__ == '__main__':
    unittest.main()
