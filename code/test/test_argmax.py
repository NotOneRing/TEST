import unittest
import torch
import numpy as np
import tensorflow as tf
from util.torch_to_tf import torch_argmax


class TestArgmax(unittest.TestCase):
    def test_argmax_dim0(self):
        """Test argmax along dimension 0 for both torch and tensorflow."""
        # PyTorch tensor
        tensor_torch = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result_torch = torch.argmax(tensor_torch, dim=0)
        
        # TensorFlow tensor
        tensor_tf = tf.convert_to_tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        result_tf = torch_argmax(tensor_tf, dim=0)
        
        # Assert results match
        self.assertTrue(np.allclose(result_torch.numpy(), result_tf.numpy()))
        
    def test_argmax_dim1(self):
        """Test argmax along dimension 1 for both torch and tensorflow."""
        # PyTorch tensor
        tensor_torch = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result_torch = torch.argmax(tensor_torch, dim=1)
        
        # TensorFlow tensor
        tensor_tf = tf.convert_to_tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        result_tf = torch_argmax(tensor_tf, dim=1)
        
        # Assert results match
        self.assertTrue(np.allclose(result_torch.numpy(), result_tf.numpy()))


if __name__ == '__main__':
    unittest.main()
