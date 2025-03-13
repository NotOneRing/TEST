import unittest
import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
from util.torch_to_tf import torch_mse_loss


class TestMSELoss(unittest.TestCase):
    def setUp(self):
        self.torch_input = torch.tensor([0.5, 1.0, 1.5])
        self.torch_target = torch.tensor([1.0, 1.0, 1.0])
        
        self.tf_input = tf.constant([0.5, 1.0, 1.5])
        self.tf_target = tf.constant([1.0, 1.0, 1.0])

    def test_mse_loss_mean(self):
        """Test MSE loss with 'mean' reduction"""
        loss_torch = F.mse_loss(self.torch_input, self.torch_target, reduction='mean')
        
        loss_tf = torch_mse_loss(self.tf_input, self.tf_target, reduction='mean')
        
        self.assertTrue(np.allclose(loss_tf.numpy(), loss_torch.numpy()))

    def test_mse_loss_sum(self):
        """Test MSE loss with 'sum' reduction"""
        loss_torch = F.mse_loss(self.torch_input, self.torch_target, reduction='sum')
        
        loss_tf = torch_mse_loss(self.tf_input, self.tf_target, reduction='sum')
        
        self.assertTrue(np.allclose(loss_tf.numpy(), loss_torch.numpy()))

    def test_mse_loss_none(self):
        """Test MSE loss with 'none' reduction"""
        loss_torch = F.mse_loss(self.torch_input, self.torch_target, reduction='none')
        
        loss_tf = torch_mse_loss(self.tf_input, self.tf_target, reduction='none')
        
        self.assertTrue(np.allclose(loss_tf.numpy(), loss_torch.numpy()))


if __name__ == '__main__':
    unittest.main()
