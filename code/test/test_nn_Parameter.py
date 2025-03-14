import unittest
import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import nn_Parameter, torch_exp

class TestNNParameter(unittest.TestCase):
    def setUp(self):
        # Define test data for both PyTorch and TensorFlow
        self.data_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.data_tf = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        
        # Create parameters
        self.param_torch = torch.nn.Parameter(self.data_torch, requires_grad=True)
        self.param_tf = nn_Parameter(self.data_tf, requires_grad=True)

    def test_parameter_creation(self):
        """Test that parameters are created correctly with proper attributes"""
        # PyTorch Parameter
        self.assertIsInstance(self.param_torch, torch.nn.Parameter)
        self.assertTrue(self.param_torch.requires_grad)
        
        # TensorFlow Parameter
        self.assertIsInstance(self.param_tf, tf.Variable)
        self.assertTrue(self.param_tf.trainable)
    
    def test_parameter_data(self):
        """Test that parameter data is the same between PyTorch and TensorFlow"""
        torch_param_data = self.param_torch.data.numpy()
        tf_param_data = self.param_tf.numpy()
        
        # Check if data is the same
        self.assertTrue(np.allclose(torch_param_data, tf_param_data))
    
    def test_requires_grad(self):
        """Test that requires_grad/trainable property is the same"""
        torch_requires_grad = self.param_torch.requires_grad
        tf_requires_grad = self.param_tf.trainable
        
        # Check if requires_grad/trainable is the same
        self.assertEqual(torch_requires_grad, tf_requires_grad)
    
    def test_torch_exp_compatibility(self):
        """Test that torch_exp works with the Parameter"""
        # Apply torch_exp to the parameter
        result = torch_exp(self.param_tf)
        
        # print("result = ", result)
        self.assertIsInstance(result, tf.Tensor)

        # Verify the exponential operation was applied correctly
        expected = np.exp(self.data_tf.numpy())
        actual = result.numpy()
        self.assertTrue(np.allclose(expected, actual))

if __name__ == '__main__':
    unittest.main()
