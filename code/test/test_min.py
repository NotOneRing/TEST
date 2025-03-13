import unittest
import torch
import numpy as np
import tensorflow as tf
from util.torch_to_tf import torch_min


class TestMin(unittest.TestCase):
    def setUp(self):
        # Define the test array and tensors
        self.x = np.array([[1, 2, 3],
                           [4, 0, 6],
                           [7, 8, 9]])
        self.torch_x = torch.tensor(self.x)
        self.tf_x = tf.convert_to_tensor(self.x)

    def test_torch_min_dim(self):
        """Test PyTorch min function with dimension parameter"""
        min_result = self.torch_x.min(dim=0)
        
        expected_values = torch.tensor([1, 0, 3])
        expected_indices = torch.tensor([0, 1, 0])
        
        self.assertTrue(torch.all(torch.eq(min_result.values, expected_values)))
        self.assertTrue(torch.all(torch.eq(min_result.indices, expected_indices)))

        tf_min_result = torch_min(self.tf_x, dim=0)
        
        expected_values = tf.constant([1, 0, 3])
        expected_indices = tf.constant([0, 1, 0])

        # print("tf_min_result.values.dtype = ", tf_min_result.values.dtype)
        # print("tf_min_result.indices.dtype = ", tf_min_result.indices.dtype)
        
        self.assertTrue(tf.reduce_all(tf.equal( tf.cast(tf_min_result.values, tf.int32), expected_values)))
        self.assertTrue(tf.reduce_all(tf.equal( tf.cast(tf_min_result.indices, tf.int32), expected_indices)))

        self.assertTrue( np.allclose(tf_min_result.values, min_result.values) )
        self.assertTrue( np.allclose(tf_min_result.indices, min_result.indices) )

    def test_torch_min_global(self):
        """Test PyTorch min function without dimension parameter (global min)"""
        min_result = self.torch_x.min()
        
        self.assertEqual(min_result.item(), 0)

        tf_min_result = torch_min(self.tf_x)
        
        self.assertEqual(tf_min_result.numpy(), 0)

        self.assertTrue( np.allclose( min_result.numpy(), tf_min_result.numpy() ) )

if __name__ == '__main__':
    unittest.main()
