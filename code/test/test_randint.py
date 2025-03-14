import torch
import tensorflow as tf
import numpy as np
import unittest

from util.torch_to_tf import torch_randint

class TestRandint(unittest.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        torch.manual_seed(42)
        tf.random.set_seed(42)
    
    def test_randint(self):
        # PyTorch randint
        low, high = 0, 100
        size = (2, 3)
        torch_tensor = torch.randint(low, high, size)

        # TensorFlow randint
        tf_tensor = torch_randint(low=low, high=high, size=size)

        # # Compare outputs
        # print("Torch tensor:\n", torch_tensor.numpy())
        # print("TensorFlow tensor:\n", tf_tensor.numpy())
        
        self.assertEqual(torch_tensor.shape, tf_tensor.shape)

# Run the test
if __name__ == "__main__":
    unittest.main()
