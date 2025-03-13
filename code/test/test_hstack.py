import unittest
import torch
import tensorflow as tf
import numpy as np
from util.torch_to_tf import torch_hstack, torch_tensor_view


class TestHstack(unittest.TestCase):
    def test_hstack(self):
        # PyTorch tensors
        a_torch = torch.tensor([[1, 2], [3, 4]]).reshape(1, 1, 4)
        b_torch = torch.tensor([[5, 6], [7, 8]]).reshape(1, 1, 4)
        
        # Apply torch.hstack
        result_torch = torch.hstack((a_torch, b_torch))
        
        # Expected shape and values
        self.assertEqual(result_torch.shape, torch.Size([1, 2, 4]))
        
        # TensorFlow tensors
        a_tf = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        b_tf = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
        
        # Reshape using torch_tensor_view
        a_tf = torch_tensor_view(a_tf, 1, 1, 4)
        b_tf = torch_tensor_view(b_tf, 1, 1, 4)
        
        # Apply torch_hstack
        result_tf = torch_hstack((a_tf, b_tf))

        # Expected shape and values
        self.assertEqual(result_tf.shape, tf.TensorShape([1, 2, 4]))

        # Compare PyTorch and TensorFlow results
        self.assertTrue(np.allclose(result_torch.numpy(), result_tf.numpy()))


if __name__ == '__main__':
    unittest.main()
