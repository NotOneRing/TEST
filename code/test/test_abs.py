import unittest
import torch
import tensorflow as tf
import numpy as np
from util.torch_to_tf import torch_abs

class TestAbsFunction(unittest.TestCase):
    """
    Test class to compare PyTorch and TensorFlow abs function implementations.
    """
    
    def test_abs_function(self):
        """Test that PyTorch abs and TensorFlow torch_abs produce consistent results."""
        # Input values to test
        input_values = [1.0, -2.0, 3.0]
        
        # PyTorch implementation
        torch_tensor = torch.tensor(input_values)
        torch_result = torch.abs(torch_tensor)
        
        # TensorFlow implementation
        tf_tensor = tf.constant(input_values)
        tf_result = torch_abs(tf_tensor)
        
        # Convert results to numpy arrays for comparison
        torch_np = torch_result.detach().numpy()
        tf_np = tf_result.numpy()
        
        # # Print results for debugging
        # print(f"PyTorch result: {torch_result}")
        # print(f"TensorFlow result: {tf_result}")
        
        # Assert that the results are equal (within a small tolerance)
        np.testing.assert_allclose(torch_np, tf_np, rtol=1e-5, atol=1e-5)
        
        # Check specific values
        expected_values = [1.0, 2.0, 3.0]
        for i, expected in enumerate(expected_values):
            self.assertAlmostEqual(torch_np[i], expected, places=5)
            self.assertAlmostEqual(tf_np[i], expected, places=5)

if __name__ == '__main__':
    unittest.main()





























