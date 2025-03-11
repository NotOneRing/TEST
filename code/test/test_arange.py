import unittest
import numpy as np
from util.torch_to_tf import torch_arange
import torch


class TestArange(unittest.TestCase):
    def test_arange_basic(self):
        """Test basic functionality of torch_arange from 0 to 10."""
        tensor_tf1 = torch_arange(0, 10)
        tensor1 = torch.arange(0, 10)
        
        # Print for debugging purposes
        print(tensor_tf1)
        print(tensor1)
        
        # Assert that the outputs are equivalent
        self.assertTrue(np.allclose(tensor_tf1.numpy(), tensor1.numpy()))
    
    def test_arange_with_step(self):
        """Test torch_arange with a step parameter."""
        tensor_tf2 = torch_arange(1, 10, step=2)
        tensor2 = torch.arange(1, 10, step=2)
        
        # Print for debugging purposes
        print(tensor_tf2)
        print(tensor2)
        
        # Assert that the outputs are equivalent
        self.assertTrue(np.allclose(tensor_tf2.numpy(), tensor2.numpy()))


if __name__ == '__main__':
    unittest.main()
