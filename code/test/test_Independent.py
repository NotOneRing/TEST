import unittest
import torch
import tensorflow as tf
import numpy as np
from torch.distributions import Normal as TorchNormal, Independent as TorchIndependent
from util.torch_to_tf import Normal, Independent, torch_zeros, torch_ones


class TestIndependent(unittest.TestCase):
    def test_tensorflow_independent(self):
        """Test Independent distribution with TensorFlow."""
        # Define a batch of data, each data point is two independent normal distributions
        base_distribution = Normal(torch_zeros(3, 2), torch_ones(3, 2))
        
        # Use Independent, regard them as independent distributions
        independent_distribution = Independent(base_distribution, reinterpreted_batch_ndims=1)
        
        # Draw samples
        samples = independent_distribution.sample()
        
        # Verify samples shape
        self.assertEqual(samples.shape, (3, 2))
        
        # Calculate log probability
        log_prob = independent_distribution.log_prob(samples)
        
        # Verify log_prob shape
        self.assertEqual(log_prob.shape, (3,))
        
        # # Print for debugging/verification
        # print(f"TF Independent Samples: {samples}")
        # print(f"TF Log probability: {log_prob.numpy()}")
        
        return samples, log_prob  # Return for use in PyTorch test

    def test_pytorch_independent(self):
        """Test Independent distribution with PyTorch."""
        # Get samples from TensorFlow test
        tf_samples, tf_log_prob = self.test_tensorflow_independent()
        
        # Define a batch of data points in PyTorch
        base_distribution = TorchNormal(torch.zeros(3, 2), torch.ones(3, 2))
        
        # Use Independent, regard them as independent distributions
        independent_distribution = TorchIndependent(base_distribution, reinterpreted_batch_ndims=1)
        
        # Convert TensorFlow samples to PyTorch tensor
        torch_samples = torch.tensor(tf_samples.numpy())
        
        # Calculate log probability
        log_prob = independent_distribution.log_prob(torch_samples)
        
        # Verify log_prob shape
        self.assertEqual(log_prob.shape, (3,))
        
        self.assertTrue( np.allclose(log_prob.numpy(), tf_log_prob.numpy()) )

        # # Print for debugging/verification
        # print(f"PyTorch Independent Samples: {torch_samples}")
        # print(f"PyTorch Log probability: {log_prob.numpy()}")

    def test_3d_independent(self):
        """Test Independent distribution with 3D batch."""
        # Define one three-dimensional normal distribution
        base_distribution = Normal(torch_zeros(2, 3, 4), torch_ones(2, 3, 4))
        
        # Use Independent to handle the independent distribution within each batch
        independent_distribution = Independent(base_distribution, reinterpreted_batch_ndims=2)
        
        # Draw samples
        samples = independent_distribution.sample()
        
        # Verify samples shape
        self.assertEqual(samples.shape, (2, 3, 4))
        
        # Calculate log probability
        log_prob = independent_distribution.log_prob(samples)
        
        # Verify log_prob shape
        self.assertEqual(log_prob.shape, (2,))
        
        # # Print for debugging/verification
        # print(f"3D Independent Samples shape: {samples.shape}")
        # print(f"3D Log probability shape: {log_prob.shape}")


if __name__ == "__main__":
    unittest.main()
