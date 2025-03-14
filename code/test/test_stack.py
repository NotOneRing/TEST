import unittest
import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_stack

class TestStack(unittest.TestCase):
    """
    Test cases for torch_stack function which implements PyTorch's stack functionality
    for TensorFlow tensors.
    """
    
    def setUp(self):
        """
        Set up test fixtures - create TensorFlow and PyTorch tensors for testing.
        """
        # TensorFlow tensors
        self.a = tf.constant([[1, 2, 3], [4, 5, 6]])
        self.b = tf.constant([[7, 8, 9], [10, 11, 12]])
        
        # PyTorch tensors
        self.torch_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.torch_b = torch.tensor([[7, 8, 9], [10, 11, 12]])
    
    def test_stack_dim0(self):
        """
        Test stacking tensors along dimension 0.
        """
        # Stack along dimension 0
        s1 = torch_stack([self.a, self.b], dim=0)
        torch_s1 = torch.stack([self.torch_a, self.torch_b], dim=0)
        
        # Verify results match between TensorFlow and PyTorch implementations
        self.assertTrue(np.allclose(s1.numpy(), torch_s1.numpy()))
    
    def test_stack_dim1(self):
        """
        Test stacking tensors along dimension 1.
        """
        # Stack along dimension 1
        s2 = torch_stack([self.a, self.b], dim=1)
        torch_s2 = torch.stack([self.torch_a, self.torch_b], dim=1)
        
        # Verify results match between TensorFlow and PyTorch implementations
        self.assertTrue(np.allclose(s2.numpy(), torch_s2.numpy()))
    
    def test_stack_default_dim(self):
        """
        Test stacking tensors with default dimension (should be 0).
        """
        # Stack with default dimension
        s3 = torch_stack([self.a, self.b])
        torch_s3 = torch.stack([self.torch_a, self.torch_b])
        
        # Verify results match between TensorFlow and PyTorch implementations
        self.assertTrue(np.allclose(s3.numpy(), torch_s3.numpy()))
        
        # Verify shape is as expected
        self.assertEqual(s3.numpy().shape, torch_s3.numpy().shape)

if __name__ == '__main__':
    unittest.main()
