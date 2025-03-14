import unittest
import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_unravel_index

class TestUnravelIndex(unittest.TestCase):
    """Test case for unravel_index function comparison between PyTorch and TensorFlow."""
    
    def test_unravel_index(self):
        """
        Test that torch_unravel_index function correctly mimics PyTorch's unravel_index
        by comparing their outputs for the same input data.
        """
        # Input data
        shape = (4, 3, 2)
        linear_indices = [0, 5, 11, 17, 23]

        # PyTorch test
        pytorch_indices = torch.tensor(linear_indices)
        pytorch_shape = shape
        pytorch_result = torch.unravel_index(pytorch_indices, pytorch_shape)
        
        # TensorFlow test
        tf_indices = tf.constant(linear_indices, dtype=tf.int32)
        tf_shape = shape
        tf_result = torch_unravel_index(tf_indices, tf_shape)
        
        # print("tf_result = ", tf_result)
        # Compare results
        for i, (torch_res, tf_res) in enumerate(zip(pytorch_result, tf_result)):
            self.assertTrue(
                np.array_equal(torch_res.numpy(), tf_res.numpy()),
                f"Dimension {i} results do not match: PyTorch {torch_res.numpy()} vs TensorFlow {tf_res.numpy()}"
            )


if __name__ == "__main__":
    unittest.main()
