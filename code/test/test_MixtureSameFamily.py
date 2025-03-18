import unittest
import numpy as np
import torch
import tensorflow as tf
import torch.distributions as D
from util.torch_to_tf import Normal, Independent, Categorical, MixtureSameFamily


class TestMixtureSameFamily(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)
        
        self.batch_size = 4
        self.num_modes = 2
        self.dim = 3
        
        means = np.random.randn(self.batch_size, self.num_modes, self.dim)
        scales = np.abs(np.random.randn(self.batch_size, self.num_modes, self.dim)) * 0.5
        logits = np.random.randn(self.batch_size, self.num_modes)
        
        # Convert to PyTorch tensors
        self.means_torch = torch.tensor(means, dtype=torch.float32)
        self.scales_torch = torch.tensor(scales, dtype=torch.float32)
        self.logits_torch = torch.tensor(logits, dtype=torch.float32)
        
        # Convert to TensorFlow tensors
        self.means_tf = tf.convert_to_tensor(means, dtype=tf.float32)
        self.scales_tf = tf.convert_to_tensor(scales, dtype=tf.float32)
        self.logits_tf = tf.convert_to_tensor(logits, dtype=tf.float32)

    def test_log_prob_comparison(self):
        # PyTorch implementation
        # Mixture components for PyTorch
        component_distribution = D.Normal(loc=self.means_torch, scale=self.scales_torch)
        component_distribution = D.Independent(component_distribution, 1)

        # Unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=self.logits_torch)
        dist_torch = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        # Option 1: calculate average for dimension modes
        sample_torch = self.means_torch.mean(dim=1)  # calculate the average for each sample in all modes
        log_prob_torch = dist_torch.log_prob(sample_torch)

        # print("Torch Log Probability: ", log_prob_torch)

        # TensorFlow implementation
        # Mixture components for TensorFlow
        component_distribution_tf = Normal(self.means_tf, self.scales_tf)
        component_distribution_tf = Independent(component_distribution_tf, 1)

        # Unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution_tf = Categorical(logits=self.logits_tf)
        dist_tf = MixtureSameFamily(
            mixture_distribution=mixture_distribution_tf,
            component_distribution=component_distribution_tf,
        )

        # Option 1: calculate average for dimension modes
        sample_tf = tf.reduce_mean(self.means_tf, axis=1)  # calculate the average for each sample in all modes
        log_prob_tf = dist_tf.log_prob(sample_tf)

        # print("TensorFlow Log Probability: ", log_prob_tf)


        self.assertTrue(np.allclose(log_prob_torch, log_prob_tf))


if __name__ == '__main__':
    unittest.main()
