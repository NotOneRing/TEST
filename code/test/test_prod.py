import unittest
import torch
import tensorflow as tf
import numpy as np
from util.torch_to_tf import torch_prod

class TestProd(unittest.TestCase):
    def test_prod(self):
        # Test PyTorch prod function
        # create a tensor
        tensor_torch = torch.tensor([[1, 2, 3], [4, 5, 6]])

        # calculate the prod of all elements
        result0_torch = torch.prod(tensor_torch)
        # print(result0_torch)  # Keep print for debugging
        
        # calculate the prod along the dimension 0
        result1_torch = torch.prod(tensor_torch, dim=0)
        # print(result1_torch)  # Keep print for debugging
        
        # calculate the prod along the dimension 1
        result2_torch = torch.prod(tensor_torch, dim=1)
        # print(result2_torch)  # Keep print for debugging

        # Test TensorFlow prod function
        # create a tensor
        tensor_tf = tf.constant([[1, 2, 3], [4, 5, 6]])

        # calculate the prod of all elements
        result0_tf = torch_prod(tensor_tf)
        # print(result0_tf)  # Keep print for debugging
        
        # calculate the prod along the dimension 0
        result1_tf = torch_prod(tensor_tf, dim=0)
        # print(result1_tf)  # Keep print for debugging
        
        # calculate the prod along the dimension 1
        result2_tf = torch_prod(tensor_tf, dim=1)
        # print(result2_tf)  # Keep print for debugging

        # Assert that PyTorch and TensorFlow implementations give the same results
        self.assertEqual(result0_torch.item(), result0_tf.numpy())
        self.assertTrue( np.allclose(result1_torch.numpy(), result1_tf.numpy() ))
        self.assertTrue( np.allclose(result2_torch.numpy(), result2_tf.numpy() ))

if __name__ == '__main__':
    unittest.main()
