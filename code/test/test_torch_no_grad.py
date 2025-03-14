import unittest
import tensorflow as tf
from util.torch_to_tf import torch_no_grad

class TestTorchNoGrad(unittest.TestCase):
    def setUp(self):
        """Set up test variables before each test method."""
        # Create a variable for testing
        self.x = tf.Variable([1.0, 2.0, 3.0], trainable=True)
    
    def test_normal_gradient_flow(self):
        """Test normal gradient flow without torch_no_grad."""
        # Normal gradient calculation with TensorFlow's GradientTape
        with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
            y = self.x * 2
            z = y + 1
            loss = tf.reduce_sum(z)

        # Calculate gradients for each variable
        x_grad = tape.gradient(loss, self.x)
        y_grad = tape.gradient(loss, y)
        z_grad = tape.gradient(loss, z)
        
        # Assert that gradients are calculated correctly
        self.assertIsNotNone(x_grad)
        self.assertIsNotNone(y_grad)
        self.assertIsNotNone(z_grad)
        
        # Expected gradients should be:
        # x_grad = [2.0, 2.0, 2.0] (derivative of loss with respect to x)
        # y_grad = [1.0, 1.0, 1.0] (derivative of loss with respect to y)
        # z_grad = [1.0, 1.0, 1.0] (derivative of loss with respect to z)
        self.assertAllEqual(x_grad, [2.0, 2.0, 2.0])
        self.assertAllEqual(y_grad, [1.0, 1.0, 1.0])
        self.assertAllEqual(z_grad, [1.0, 1.0, 1.0])
    
    def test_torch_no_grad(self):
        """Test gradient flow with torch_no_grad context manager."""
        # Using torch_no_grad context manager to prevent gradient calculation
        with torch_no_grad() as tape:
            y = self.x * 2
            z = y + 1  # y's gradient calculation is forbidden by torch_no_grad
            loss = tf.reduce_sum(z)

        # Calculate gradients for each variable
        x_grad = tape.gradient(loss, self.x)
        y_grad = tape.gradient(loss, y)
        z_grad = tape.gradient(loss, z)
        
        # Assert that x_grad is None because z does not propagate gradient
        self.assertIsNone(x_grad)
        
        # y_grad and z_grad should still be calculated
        self.assertIsNone(y_grad)
        self.assertIsNone(z_grad)
    
    def assertAllEqual(self, a, b):
        """Helper method to assert that all elements in two tensors are equal."""
        self.assertTrue(tf.reduce_all(tf.equal(a, b)))

if __name__ == '__main__':
    unittest.main()
