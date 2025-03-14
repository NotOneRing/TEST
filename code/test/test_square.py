import unittest
import torch
import numpy as np
import tensorflow as tf

from util.torch_to_tf import torch_square

class TestSquare(unittest.TestCase):
    def test_square(self):
        # Test with torch tensor
        sigma = torch.tensor([1.0, 2.0, 3.0])
        var = torch.square(sigma)  # element-wise square
        # output: tensor([1., 4., 9.])
        
        # Test with tensorflow tensor
        sigma_tf = tf.constant([1.0, 2.0, 3.0])
        var_tf = torch_square(sigma_tf)  # element-wise square
        # output: [1. 4. 9.]
        
        # Assert that both implementations produce the same result
        self.assertTrue(np.allclose(var.numpy(), var_tf.numpy()))


if __name__ == '__main__':
    unittest.main()
