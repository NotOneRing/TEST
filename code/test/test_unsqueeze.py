import unittest
import tensorflow as tf
import torch
import numpy as np
from util.torch_to_tf import torch_unsqueeze

class TestUnsqueeze(unittest.TestCase):
    """
    Test case for comparing torch.unsqueeze and tf.expand_dims functionality.
    """
    
    def test_unsqueeze(self):
        """
        Test that torch.unsqueeze and tf.expand_dims produce equivalent results.
        This test verifies that adding a dimension to a tensor works the same way
        in both PyTorch and TensorFlow.
        """
        # Create a 3x3x3 numpy array
        x = np.array(range(27)).reshape(3, 3, 3)

        # Dimension to add
        self_event_ndims = 1

        # PyTorch implementation
        x_torch = torch.tensor(x)
        x_expands = x_torch.unsqueeze(self_event_ndims)
        
        # # Print PyTorch results
        # print("x_expands = ", x_expands)
        # print("x_expands.shape = ", x_expands.shape)

        # TensorFlow implementation
        x_tf = tf.convert_to_tensor(x)
        x_expands_tf = torch_unsqueeze(x_tf, dim=self_event_ndims)
        
        # # Print TensorFlow results
        # print("x_expands_tf = ", x_expands_tf)

        # Assert that both implementations produce the same result
        self.assertTrue(np.allclose(x_expands_tf.numpy(), x_expands.numpy()))


if __name__ == '__main__':
    unittest.main()
