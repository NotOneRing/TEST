import unittest
import tensorflow as tf
import torch
import numpy as np

from util.torch_to_tf import torch_tanh

class TestTanh(unittest.TestCase):
    """Test case for comparing torch_tanh implementation with torch.tanh."""
    
    def test_tanh_equivalence(self):
        
        # Create test input tensor in TensorFlow
        x = tf.constant([0.5, 0.1, 0.9, 1.5, -0.5, -1, -3], dtype=tf.float32)
        # Apply torch_tanh function (TensorFlow implementation)
        y = torch_tanh(x)
        
        # Convert TensorFlow tensor to PyTorch tensor
        x_torch = torch.tensor(x.numpy())
        # Apply native PyTorch tanh function
        y_torch = torch.tanh(x_torch)
        
        # Check if results contain NaN or Inf values
        y_is_nan = np.isnan(y.numpy())
        y_torch_is_nan = np.isnan(y_torch.numpy())
        
        y_is_inf = np.isinf(y.numpy())
        y_torch_is_inf = np.isinf(y_torch.numpy())
        
        # Check if the NaN and Inf parts are equivalent, respectively
        self.assertTrue(np.all(y_is_nan == y_torch_is_nan), 
                        "NaN patterns differ between implementations")
        self.assertTrue(np.all(y_is_inf == y_torch_is_inf), 
                        "Inf patterns differ between implementations")
        
        # Omit NaN and Inf, check other values
        valid_indices = ~(y_is_nan | y_is_inf)
        self.assertTrue(
            np.allclose(
                y.numpy()[valid_indices], 
                y_torch.numpy()[valid_indices], 
                atol=1e-4
            ),
            "Values differ between TensorFlow and PyTorch implementations"
        )

if __name__ == '__main__':
    unittest.main()
