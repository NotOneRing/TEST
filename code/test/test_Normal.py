import unittest
import torch
import tensorflow as tf
import numpy as np
from torch.distributions import Normal as TorchNormal
from util.torch_to_tf import Normal as TFNormal

class TestNormal(unittest.TestCase):
    def setUp(self):
        # define the mean and standard deviation
        self.mean = torch.tensor([0.0, 1.0])
        self.std = torch.tensor([1.0, 0.5])
        
        # Convert to TensorFlow tensors for TF tests
        self.tf_mean = tf.convert_to_tensor(self.mean.numpy(), dtype=np.float32)  # mean
        self.tf_std = tf.convert_to_tensor(self.std.numpy(), dtype=np.float32)  # std
        
        # Point to evaluate probability density
        self.eval_point = torch.tensor([1.0, 1.0])
        self.tf_eval_point = tf.convert_to_tensor(np.array([1.0, 1.0]), dtype=np.float32)

    def test_torch_normal_sample(self):
        """Test sampling from PyTorch Normal distribution"""
        # create a Normal distribution
        dist = TorchNormal(self.mean, self.std)
        
        # sample
        sample = dist.sample()  # generate one sample
        
        # Check that sample has the right shape
        self.assertEqual(sample.shape, self.mean.shape)
        
        # # Print for debugging/verification
        # print(f"PyTorch Sample: {sample}")

    def test_torch_normal_log_prob(self):
        """Test log probability calculation in PyTorch Normal distribution"""
        # create a Normal distribution
        dist = TorchNormal(self.mean, self.std)
        
        # calculate the probability density function(PDF) value
        log_prob = dist.log_prob(self.eval_point)  # probability density function corresponding to [1.0, 1.0]
        
        # Check that log_prob has the right shape
        self.assertEqual(log_prob.shape, self.mean.shape)
        
        # # Print for debugging/verification
        # print(f"PyTorch Log probability: {log_prob}")

    def test_tf_normal_sample(self):
        """Test sampling from TensorFlow Normal distribution"""
        # create a Normal distribution
        dist = TFNormal(self.tf_mean, self.tf_std)
        
        # sample
        sample = dist.sample()  # generate one sample
        
        # Check that sample has the right shape
        self.assertEqual(sample.shape, self.tf_mean.shape)
        
        # # Print for debugging/verification
        # print(f"TensorFlow Sample: {sample}")

    def test_tf_normal_log_prob(self):
        """Test log probability calculation in TensorFlow Normal distribution"""
        # create a Normal distribution
        dist = TFNormal(self.tf_mean, self.tf_std)
        
        # calculate the probability density function(PDF) value
        log_prob = dist.log_prob(self.tf_eval_point)  # probability density corresponding to x
        
        # Check that log_prob has the right shape
        self.assertEqual(log_prob.shape, self.tf_mean.shape)
        
        # # Print for debugging/verification
        # print(f"TensorFlow Log probability at x = {self.tf_eval_point}: {log_prob.numpy()}")
        # print("log_prob = ", log_prob)
        # print("type(log_prob) = ", log_prob)

    def test_equal_log_prob(self):

        dist = TorchNormal(self.mean, self.std)
        torch_log_prob = dist.log_prob( torch.tensor(self.eval_point) )
        
        dist = TFNormal(self.tf_mean, self.tf_std)
        tf_log_prob = dist.log_prob(self.tf_eval_point)
        
        self.assertTrue( np.allclose(torch_log_prob.numpy(), tf_log_prob.numpy(), atol=1e-6) )

if __name__ == '__main__':
    unittest.main()
