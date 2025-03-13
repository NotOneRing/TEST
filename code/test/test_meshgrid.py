import unittest
import torch
import numpy as np
import tensorflow as tf
from util.torch_to_tf import torch_meshgrid


class TestMeshgrid(unittest.TestCase):
    def test_meshgrid_with_individual_tensors(self):
        # PyTorch tensors
        x_torch = torch.tensor([1, 2, 3])
        y_torch = torch.tensor([4, 5])
        
        x_tf = tf.constant([1, 2, 3])
        y_tf = tf.constant([4, 5])
        
        xx1_torch, yy1_torch = torch.meshgrid(x_torch, y_torch, indexing='ij')
        
        xx1_tf, yy1_tf = torch_meshgrid(x_tf, y_tf, indexing="ij")
        
        # Assert outputs are equal
        self.assertTrue(np.allclose(xx1_torch.numpy(), xx1_tf.numpy()))
        self.assertTrue(np.allclose(yy1_torch.numpy(), yy1_tf.numpy()))
    
    def test_meshgrid_with_tensor_list(self):
        # PyTorch tensors
        x_torch = torch.tensor([1, 2, 3])
        y_torch = torch.tensor([4, 5])
        
        # TensorFlow tensors
        x_tf = tf.constant([1, 2, 3])
        y_tf = tf.constant([4, 5])
        
        # PyTorch meshgrid with list input
        xx2_torch, yy2_torch = torch.meshgrid([x_torch, y_torch], indexing='ij')
        
        # TensorFlow meshgrid using torch_meshgrid with list input
        xx2_tf, yy2_tf = torch_meshgrid([x_tf, y_tf], indexing="ij")
        
        # Assert outputs are equal
        self.assertTrue(np.allclose(xx2_torch.numpy(), xx2_tf.numpy()))
        self.assertTrue(np.allclose(yy2_torch.numpy(), yy2_tf.numpy()))

    def test_meshgrid_with_individual_tensors_index_xy(self):
        # PyTorch tensors
        x_torch = torch.tensor([1, 2, 3])
        y_torch = torch.tensor([4, 5])
        
        x_tf = tf.constant([1, 2, 3])
        y_tf = tf.constant([4, 5])
        
        xx1_torch, yy1_torch = torch.meshgrid(x_torch, y_torch, indexing='xy')
        
        xx1_tf, yy1_tf = torch_meshgrid(x_tf, y_tf, indexing="xy")
        
        # Assert outputs are equal
        self.assertTrue(np.allclose(xx1_torch.numpy(), xx1_tf.numpy()))
        self.assertTrue(np.allclose(yy1_torch.numpy(), yy1_tf.numpy()))
    
    def test_meshgrid_with_tensor_list_index_xy(self):
        # PyTorch tensors
        x_torch = torch.tensor([1, 2, 3])
        y_torch = torch.tensor([4, 5])
        
        # TensorFlow tensors
        x_tf = tf.constant([1, 2, 3])
        y_tf = tf.constant([4, 5])
        
        # PyTorch meshgrid with list input
        xx2_torch, yy2_torch = torch.meshgrid([x_torch, y_torch], indexing='xy')
        
        # TensorFlow meshgrid using torch_meshgrid with list input
        xx2_tf, yy2_tf = torch_meshgrid([x_tf, y_tf], indexing="xy")
        
        # Assert outputs are equal
        self.assertTrue(np.allclose(xx2_torch.numpy(), xx2_tf.numpy()))
        self.assertTrue(np.allclose(yy2_torch.numpy(), yy2_tf.numpy()))


if __name__ == '__main__':
    unittest.main()
