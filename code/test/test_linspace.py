import tensorflow as tf
import torch
import unittest
import numpy as np
from util.torch_to_tf import torch_linspace

class TestLinspace(unittest.TestCase):
    def test_linspace(self):
        h = 5
        pad = 10
        eps = 1.0 / (h + pad)

        # Create torch linspace array
        torch_arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * pad
        )[:h]

        # Create tensorflow linspace array using custom implementation
        tf_arange = torch_linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad)[:h]

        # Assert that both arrays are close to each other
        self.assertTrue(np.allclose(torch_arange.numpy(), tf_arange.numpy()))

if __name__ == '__main__':
    unittest.main()
