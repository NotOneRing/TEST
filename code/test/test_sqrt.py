import unittest
import torch
import numpy as np
import tensorflow as tf

from util.torch_to_tf import torch_sqrt

class TestSqrt(unittest.TestCase):
    """Test case for the square root function implementation."""
    
    def test_sqrt(self):
        """Test that torch_sqrt function produces the same results as torch.sqrt."""
        # Create a PyTorch tensor
        sigma = torch.tensor([1.0, 2.0, 3.0])
        # Apply PyTorch's sqrt function
        var = torch.sqrt(sigma)
        # print(var)  
        
        # Create an equivalent TensorFlow tensor
        sigma = tf.constant([1.0, 2.0, 3.0])
        # Apply our torch_sqrt implementation
        var_tf = torch_sqrt(sigma)
        # print(var) 
        
        # Assert that both implementations produce the same result
        self.assertTrue(np.allclose(var.numpy(), var_tf.numpy()))


if __name__ == '__main__':
    unittest.main()
