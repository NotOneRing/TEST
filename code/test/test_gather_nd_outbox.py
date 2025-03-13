import unittest
import tensorflow as tf
import numpy as np

from util.torch_to_tf import safe_gather_nd


class TestSafeGatherND(unittest.TestCase):
    """Test cases for the safe_gather_nd function."""
    
    def test_basic_functionality(self):
        """Test basic functionality with valid indices."""
        tensor = tf.constant([[5, 6], [7, 8]])  # shape: (2,2)
        indices = tf.constant([[0, 0], [1, 1]])
        
        result = safe_gather_nd(tensor, indices)
        expected = np.array([5, 8])
        
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_out_of_bounds_indices(self):
        """Test with out-of-bounds indices."""
        tensor = tf.constant([[5, 6], [7, 8]])  # shape: (2,2)
        indices = tf.constant([[0, 0], [1, 1], [2, 2]])  # last index is out of bound
        
        result = safe_gather_nd(tensor, indices)
        expected = np.array([5, 8, 0])  # Default value is 0.0
        
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_negative_indices(self):
        """Test with negative indices."""
        tensor = tf.constant([[5, 6], [7, 8]])  # shape: (2,2)
        indices = tf.constant([[0, 0], [-1, 0]])  # negative index is out of bound
        
        result = safe_gather_nd(tensor, indices)
        expected = np.array([5, 0])  # Default value is 0.0
        
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_custom_default_value(self):
        """Test with a custom default value."""
        tensor = tf.constant([[5, 6], [7, 8]])  # shape: (2,2)
        indices = tf.constant([[0, 0], [1, 1], [2, 2]])  # last index is out of bound
        default_value = 999.0
        
        result = safe_gather_nd(tensor, indices, default_value=default_value)
        expected = np.array([5, 8, 999])
        
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_higher_dimensional_tensor(self):
        """Test with a higher dimensional tensor."""
        tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # shape: (2,2,2)
        indices = tf.constant([[0, 0, 0], [1, 1, 1], [2, 0, 0]])  # last index's first dimension is out of bound
        
        result = safe_gather_nd(tensor, indices)
        expected = np.array([1, 8, 0])
        
        np.testing.assert_array_equal(result.numpy(), expected)


if __name__ == '__main__':
    unittest.main()
