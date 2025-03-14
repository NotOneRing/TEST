import unittest
import tensorflow as tf
import numpy as np

class TestTensorShape(unittest.TestCase):
    """Test cases for TensorFlow's TensorShape functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a 3x3 array with values from 1 to 9
        self.a = np.array(range(1, 10)).reshape(3, 3)
        self.a_tensor = tf.convert_to_tensor(self.a)
        self.a_shape = self.a.shape
        self.zero_shape = tf.TensorShape([])
    
    def test_shape_properties(self):
        """Test basic properties of TensorShape objects."""
        # Test that shapes are correctly created
        self.assertEqual(self.a_shape, (3, 3))
        self.assertEqual(self.zero_shape.rank, 0)
        self.assertEqual(list(self.zero_shape.as_list()), [])
    
    def test_shape_addition(self):
        """Test addition operation between TensorShape objects."""
        # Test addition of shapes
        combined_shape = self.a_shape + self.zero_shape
        
        # Addition with zero shape should return the original shape
        self.assertEqual(combined_shape, self.a_shape)
        
        # Verify the dimensions are preserved
        self.assertEqual(combined_shape, (3, 3))
    
    def test_tensor_shape_conversion(self):
        """Test conversion between tensor and its shape."""
        # Get shape from tensor
        tensor_shape = self.a_tensor.shape
        
        # Verify tensor shape matches the original array shape
        self.assertEqual(tensor_shape, self.a_shape)
        self.assertEqual(tensor_shape.as_list(), [3, 3])

if __name__ == '__main__':
    unittest.main()
