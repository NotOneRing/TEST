import unittest
import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_ones


class TestTorchOnes(unittest.TestCase):
    """Test class for comparing ones tensor creation in PyTorch and TensorFlow."""
    def test_ones_with_tuple_shape(self):
        """Test ones tensor creation with shape passed as a tuple."""
        # Set testing shape
        shape = (2, 3)
        
        # TensorFlow: ones tensor
        tf_tensor = torch_ones(shape, dtype=tf.float32)
        
        # PyTorch: ones tensor
        torch_tensor = torch.ones(shape, dtype=torch.float32)
        
        # Check the output of tensorflow and torch are equivalent
        self.assertTrue(np.allclose(tf_tensor.numpy(), torch_tensor.numpy()))
    
    def test_ones_with_unpacked_shape(self):
        """Test ones tensor creation with shape passed as unpacked arguments."""
        # Set testing shape
        shape = (2, 3)
        
        # TensorFlow ones tensor
        tf_tensor = torch_ones(*shape, dtype=tf.float32)
        
        # PyTorch ones tensor
        torch_tensor = torch.ones(*shape, dtype=torch.float32)
        
        # Check if their output are equivalent
        self.assertTrue(np.allclose(tf_tensor.numpy(), torch_tensor.numpy()))


if __name__ == '__main__':
    unittest.main()
