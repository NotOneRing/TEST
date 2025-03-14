import unittest
import tensorflow as tf


class TestTrainableVariables(unittest.TestCase):
    """Test case for TensorFlow model trainable variables functionality."""
    
    def setUp(self):
        """Create a simple model for testing."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, input_shape=(3,), activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        
        # Get the trainable variables
        self.trainable_vars = self.model.trainable_variables
    
    def test_trainable_vars_type(self):
        """Test the type of trainable_variables collection."""
        # Check that trainable_vars is a list
        self.assertIsInstance(self.trainable_vars, list)
    
    def test_trainable_vars_content(self):
        """Test the content and properties of individual trainable variables."""
        # There should be 4 trainable variables (weights and biases for 2 layers)
        self.assertEqual(len(self.trainable_vars), 4)
        
        # Check each variable's type and properties
        for var in self.trainable_vars:
            # # Each variable should be a tf.Variable
            # print("type(var) = ", type(var))
            # self.assertIsInstance(var, tf.Variable)
            
            # Each variable should have a name, shape, and trainable attribute
            self.assertIsNotNone(var.name)
            self.assertIsNotNone(var.shape)
            self.assertTrue(var.trainable)
    
    def test_variable_shapes(self):
        """Test the shapes of the trainable variables."""
        # Expected shapes for the variables in our model:
        # Layer 1: weights (3, 4), biases (4,)
        # Layer 2: weights (4, 2), biases (2,)
        expected_shapes = [(3, 4), (4,), (4, 2), (2,)]
        
        actual_shapes = [var.shape for var in self.trainable_vars]
        for expected, actual in zip(expected_shapes, actual_shapes):
            self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
