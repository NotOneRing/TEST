import unittest
import tensorflow as tf
import torch
import numpy as np
from util.torch_to_tf import torch_atanh


class TestAtanh(unittest.TestCase):
    def test_atanh_equivalence(self):
        """Test that torch_atanh produces equivalent results to torch.atanh, handling NaN and Inf values."""
        # Create test data with values that will produce regular results, NaN, and Inf
        x = tf.constant([0.5, 0.1, 0.9, 1.5, -0.5, -1, -3], dtype=tf.float32)
        
        # Apply torch_atanh (TensorFlow implementation)
        y = torch_atanh(x)
        # print(y)
        
        # Apply torch.atanh (PyTorch implementation)
        x_torch = torch.tensor(x.numpy())
        y_torch = torch.atanh(x_torch)
        # print(y_torch)
        
        # Identify NaN and Inf values in both results
        y_is_nan = np.isnan(y.numpy())
        y_torch_is_nan = np.isnan(y_torch.numpy())

        # print(y_is_nan)
        # print(y_torch_is_nan)

        y_is_inf = np.isinf(y.numpy())
        y_torch_is_inf = np.isinf(y_torch.numpy())

        # print(y_is_inf)
        # print(y_torch_is_inf)

        # First check if NaN and Inf patterns match between implementations
        self.assertTrue(np.all(y_is_nan == y_torch_is_nan), 
                        "NaN patterns don't match between TensorFlow and PyTorch implementations")
        self.assertTrue(np.all(y_is_inf == y_torch_is_inf), 
                        "Inf patterns don't match between TensorFlow and PyTorch implementations")
        
        # Then check if the regular values are close enough
        mask = ~(y_is_nan | y_is_inf)  # Mask to exclude NaN and Inf values

        # print(mask)
        # print("y.numpy()[mask] = ", y.numpy()[mask])
        # print("y_torch.numpy()[mask] = ", y_torch.numpy()[mask])

        self.assertTrue(np.allclose(y.numpy()[mask], y_torch.numpy()[mask], atol=1e-4),
                        "Regular values don't match between TensorFlow and PyTorch implementations")


if __name__ == '__main__':
    unittest.main()
