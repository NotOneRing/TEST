import unittest
import torch
import numpy as np
import tensorflow as tf
from util.torch_to_tf import torch_max


class TestMax(unittest.TestCase):
    def setUp(self):
        # Set up test data
        self.x = np.array([[1, 2, 3],
                           [4, 0, 6],
                           [7, 8, 9]]).astype(np.int32)
        self.torch_x = torch.tensor(self.x)
        self.tf_x = tf.convert_to_tensor(self.x)

    def test_torch_max_dim(self):
        """Test PyTorch max along dimension 0"""
        max_result = self.torch_x.max(dim=0)
        
        # Verify the result has values and indices attributes
        self.assertTrue(hasattr(max_result, 'values'))
        self.assertTrue(hasattr(max_result, 'indices'))
        
        # Verify the shapes are correct
        self.assertEqual(max_result.values.shape, torch.Size([3]))
        self.assertEqual(max_result.indices.shape, torch.Size([3]))
        
        # Verify the values are correct
        expected_values = torch.tensor([7, 8, 9])
        expected_indices = torch.tensor([2, 2, 2])


        # print("torch: expected_values.dtype = ", expected_values.dtype)
        # print("torch: expected_indices.dtype = ", expected_indices.dtype)
        # print("torch: max_result.values.dtype = ", max_result.values.dtype)
        # print("torch: max_result.indices.dtype = ", max_result.indices.dtype)


        self.assertTrue(torch.all(torch.eq(max_result.values, expected_values)))
        self.assertTrue(torch.all(torch.eq(max_result.indices, expected_indices)))

    def test_torch_max_global(self):
        """Test PyTorch global max"""
        max_result = self.torch_x.max()
        
        # Verify the result is a tensor
        self.assertIsInstance(max_result, torch.Tensor)
        
        # Verify the value is correct
        self.assertEqual(max_result.item(), 9)

    def test_tf_max_dim(self):
        """Test TensorFlow max along dimension 0"""
        max_result = torch_max(self.tf_x, dim=0)
        
        # Verify the result has values and indices attributes
        self.assertTrue(hasattr(max_result, 'values'))
        self.assertTrue(hasattr(max_result, 'indices'))
        
        # Verify the shapes are correct
        self.assertEqual(max_result.values.shape, (3,))
        self.assertEqual(max_result.indices.shape, (3,))
        
        # Verify the values are correct
        expected_values = tf.constant([7, 8, 9])
        expected_indices = tf.constant([2, 2, 2])

        # print("tf: expected_values.dtype = ", expected_values.dtype)
        # print("tf: expected_indices.dtype = ", expected_indices.dtype)
        # print("tf: max_result.values.dtype = ", max_result.values.dtype)
        # print("tf: max_result.indices.dtype = ", max_result.indices.dtype)

        self.assertTrue(tf.reduce_all(tf.equal(max_result.values, expected_values)))
        self.assertTrue(tf.reduce_all(tf.equal( tf.cast(max_result.indices, tf.int32), expected_indices)))

    def test_tf_max_global(self):
        """Test TensorFlow global max"""
        max_result = torch_max(self.tf_x)
        
        # Verify the result is a tensor
        self.assertIsInstance(max_result, tf.Tensor)
        
        # Verify the value is correct
        self.assertEqual(max_result.numpy(), 9)


if __name__ == '__main__':
    unittest.main()
