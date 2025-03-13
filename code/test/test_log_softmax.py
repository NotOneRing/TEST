import unittest
import torch
import tensorflow as tf
import numpy as np


class TestLogSoftmax(unittest.TestCase):
    def test_torch_basic_log_softmax(self):
        """Test basic PyTorch log_softmax on a 1D tensor."""
        x = torch.tensor([1.0, 2.0, 3.0])
        log_softmax_x = torch.log_softmax(x, dim=0)
        
        expected = torch.tensor([-2.4076, -1.4076, -0.4076])
        self.assertTrue(torch.allclose(log_softmax_x, expected, rtol=1e-4))
    
    def test_torch_manual_vs_direct_log_softmax(self):
        """Test that manual calculation of log_softmax matches the direct function."""
        x = torch.tensor([1.0, 2.0, 3.0])
        
        softmax = torch.exp(x) / torch.sum(torch.exp(x))
        log_softmax_manual = torch.log(softmax)
        
        log_softmax_direct = torch.log_softmax(x, dim=0)
        
        self.assertTrue(torch.allclose(log_softmax_manual, log_softmax_direct, rtol=1e-5))
    
    def test_torch_2d_log_softmax(self):
        """Test PyTorch log_softmax on a 2D tensor with dim=1."""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        log_softmax_x = torch.log_softmax(x, dim=1)
        
        row1 = torch.tensor([1.0, 2.0, 3.0])
        row2 = torch.tensor([4.0, 5.0, 6.0])
        
        expected_row1 = torch.log_softmax(row1, dim=0)
        expected_row2 = torch.log_softmax(row2, dim=0)
        expected = torch.stack([expected_row1, expected_row2])
        
        self.assertTrue(torch.allclose(log_softmax_x, expected, rtol=1e-5))
    
    def test_tf_basic_log_softmax(self):
        """Test basic TensorFlow log_softmax on a 1D tensor."""
        x = tf.constant([1.0, 2.0, 3.0])
        log_softmax_x = tf.nn.log_softmax(x)
        
        expected = tf.constant([-2.4076059, -1.4076059, -0.4076059])
        self.assertTrue(np.allclose(log_softmax_x.numpy(), expected.numpy(), rtol=1e-5))
    
    def test_tf_manual_vs_direct_log_softmax(self):
        """Test that manual calculation of log_softmax matches the direct function in TensorFlow."""
        x = tf.constant([1.0, 2.0, 3.0])
        
        softmax = tf.exp(x) / tf.reduce_sum(tf.exp(x))
        log_softmax_manual = tf.math.log(softmax)
        
        log_softmax_direct = tf.nn.log_softmax(x)
        
        self.assertTrue(np.allclose(log_softmax_manual.numpy(), log_softmax_direct.numpy(), rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
