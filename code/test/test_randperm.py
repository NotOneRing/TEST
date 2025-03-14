import unittest
import torch
import tensorflow as tf

from util.torch_to_tf import torch_randperm
from util.config import DEBUG


class TestRandperm(unittest.TestCase):
    def test_randperm(self):
        # Test PyTorch implementation
        # create an integer sequence from 0 to 9 and shuffle it randomly
        tensor = torch.randperm(10)
        # print(tensor)
        
        # Verify PyTorch output
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (10,))
        self.assertEqual(set(tensor.tolist()), set(range(10)))
        
        # Test TensorFlow implementation
        # create an integer sequence from 0 to 9 and shuffle it randomly
        tensor_tf = torch_randperm(10)
        # print("type(tensor_tf)", type(tensor_tf))
        # print(tensor_tf)
        
        # Verify TensorFlow output
        self.assertIsInstance(tensor_tf, tf.Tensor)
        self.assertEqual(tensor_tf.shape, (10,))
        self.assertEqual(set(tensor_tf.numpy().tolist()), set(range(10)))


if __name__ == "__main__":
    unittest.main()
