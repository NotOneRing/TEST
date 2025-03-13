import tensorflow as tf
import unittest
import numpy as np


class TestGatherND(unittest.TestCase):
    def setUp(self):
        # Create the tensor for all tests
        self.tensor = tf.range(0, 27, 1)
        self.tensor = tf.reshape(self.tensor, (3, 3, 3))
    
    def test_explicit_indices(self):
        """Test gather_nd with explicitly defined indices"""
        # Define indices list
        indices = tf.constant([[0, 0], [1, 1], [2, 2]])
        
        # Use tf.gather_nd
        result = tf.gather_nd(self.tensor, indices)
        
        # Convert to numpy for assertion
        result_np = result.numpy()
        
        # Expected values based on the original test
        expected = np.array([self.tensor[0, 0].numpy(), 
                             self.tensor[1, 1].numpy(), 
                             self.tensor[2, 2].numpy()])
        
        # Assert that the results match
        np.testing.assert_array_equal(result_np, expected)
    
    def test_stacked_indices(self):
        """Test gather_nd with indices created by stacking rows and columns"""
        # Define indices for rows and columns
        rows = tf.constant([0, 1, 2])  # index along 0 dim
        cols = tf.constant([0, 1, 2])  # index along 1 dim
        
        # Use tf.stack to create indices
        indices = tf.stack([rows, cols], axis=1)
        
        # Use tf.gather_nd to get values
        result = tf.gather_nd(self.tensor, indices)
        
        # Convert to numpy for assertion
        result_np = result.numpy()
        
        # Expected values based on the original test
        expected = np.array([self.tensor[0, 0].numpy(), 
                             self.tensor[1, 1].numpy(), 
                             self.tensor[2, 2].numpy()])
        
        # Assert that the results match
        np.testing.assert_array_equal(result_np, expected)


if __name__ == '__main__':
    unittest.main()
