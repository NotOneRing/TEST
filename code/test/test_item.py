import unittest
import torch
import numpy as np
import tensorflow as tf
from util.func_pytest import np_to_tf, np_to_torch
from util.torch_to_tf import torch_tensor_item


class TestItem(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures, called before every test method."""
        self.arr = np.array([1.0])
        self.tf_arr = np_to_tf(self.arr)
        self.torch_arr = np_to_torch(self.arr)

    def test_torch_item(self):
        """Test that torch tensor's item() method returns the expected value and type."""
        self.assertEqual(self.torch_arr.item(), 1.0)
        
        self.assertIsInstance(self.torch_arr.item(), float)
        
        # print("torch_arr.item() = ", self.torch_arr.item())
        # print("type(torch_arr.item()) = ", type(self.torch_arr.item()))

    def test_torch_tensor_item(self):
        """Test that torch_tensor_item function works correctly on tensorflow tensors."""
        result = torch_tensor_item(self.tf_arr)
        
        self.assertEqual(result, 1.0)
        
        self.assertIsInstance(result, float)
        
        # print("torch_tensor_item(tf_arr)", result)
        # print("type(torch_tensor_item(tf_arr))", type(result))


if __name__ == '__main__':
    unittest.main()
