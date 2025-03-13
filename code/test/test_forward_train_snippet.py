import unittest
import numpy as np
import torch
import tensorflow as tf
from util.torch_to_tf import Normal, Independent, Categorical, MixtureSameFamily
import torch.distributions as D


class TestForwardTrainSnippet(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define common parameters
        self.batch_size = 4
        self.num_modes = 2
        self.dim = 3
        
        # Step 1: use numpy to construct data
        self.means_np = np.random.randn(self.batch_size, self.num_modes, self.dim)
        self.scales_np = np.abs(np.random.randn(self.batch_size, self.num_modes, self.dim)) * 0.5
        self.logits_np = np.random.randn(self.batch_size, self.num_modes)

        # Generate a sample x for log_prob
        self.x_np = np.random.randn(self.batch_size, self.dim)
        self.x_tf = tf.convert_to_tensor( self.x_np , dtype=tf.float32)
        self.x_torch = torch.tensor( self.x_np , dtype=torch.float32)

        # Step 2: convert numpy into torch and tensorflow
        self.means_torch = torch.tensor(self.means_np, dtype=torch.float32)
        self.scales_torch = torch.tensor(self.scales_np, dtype=torch.float32)
        self.logits_torch = torch.tensor(self.logits_np, dtype=torch.float32)
        
        self.means_tf = tf.convert_to_tensor(self.means_np, dtype=tf.float32)
        self.scales_tf = tf.convert_to_tensor(self.scales_np, dtype=tf.float32)
        self.logits_tf = tf.convert_to_tensor(self.logits_np, dtype=tf.float32)

    def test_torch_distributions(self):
        """Test PyTorch distribution functionality"""
        means = self.means_torch
        scales = self.scales_torch
        logits = self.logits_torch
        
        # Create component distribution
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)
        
        # Calculate entropy
        component_entropy = component_distribution.entropy()
        approx_entropy = torch.mean(
            torch.sum(logits.softmax(-1) * component_entropy, dim=-1)
        )
        std = torch.mean(torch.sum(logits.softmax(-1) * scales.mean(-1), dim=-1))
        
        # Create mixture distribution
        mixture_distribution = D.Categorical(logits=logits)
        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        
        # Verify results are not None and have expected shapes
        self.assertIsNotNone(std)
        self.assertIsNotNone(approx_entropy)
        
        log_prob = dist.log_prob(self.x_torch)

        self.assertIsNotNone(log_prob)
        
        # # Print values for debugging (can be removed in production)
        # print("PyTorch std =", std)
        # print("PyTorch approx_entropy =", approx_entropy)
        # print("PyTorch dist.log_prob() =", log_prob)

    def test_tensorflow_distributions(self):
        """Test TensorFlow distribution functionality"""
        means = self.means_tf
        scales = self.scales_tf
        logits = self.logits_tf
        
        # Create component distribution
        component_distribution = Normal(means, scales)
        component_distribution = Independent(component_distribution, 1)
        
        # Calculate entropy
        component_entropy = component_distribution.entropy()
        approx_entropy = tf.reduce_mean(
            tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * component_entropy, axis=-1)
        )
        std = tf.reduce_mean(
            tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * tf.reduce_mean(scales, axis=-1), axis=-1)
        )
        
        # Create mixture distribution
        mixture_distribution = Categorical(logits=logits)
        dist = MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        
        # Verify results are not None and have expected shapes
        self.assertIsNotNone(std)
        self.assertIsNotNone(approx_entropy)
        
        log_prob = dist.log_prob(self.x_tf)

        self.assertIsNotNone(log_prob)
        
        # # Print values for debugging (can be removed in production)
        # print("TensorFlow std =", std)
        # print("TensorFlow approx_entropy =", approx_entropy)
        # print("TensorFlow dist.log_prob() =", log_prob)

    def test_compare_torch_and_tensorflow(self):
        """Compare results between PyTorch and TensorFlow implementations"""
        # PyTorch calculations
        means_torch = self.means_torch
        scales_torch = self.scales_torch
        logits_torch = self.logits_torch
        
        component_distribution_torch = D.Normal(loc=means_torch, scale=scales_torch)
        component_distribution_torch = D.Independent(component_distribution_torch, 1)
        
        component_entropy_torch = component_distribution_torch.entropy()
        approx_entropy_torch = torch.mean(
            torch.sum(logits_torch.softmax(-1) * component_entropy_torch, dim=-1)
        )
        std_torch = torch.mean(torch.sum(logits_torch.softmax(-1) * scales_torch.mean(-1), dim=-1))
        
        mixture_distribution_torch = D.Categorical(logits=logits_torch)
        dist_torch = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution_torch,
            component_distribution=component_distribution_torch,
        )

        log_prob_torch = dist_torch.log_prob(self.x_torch)
        
        # TensorFlow calculations
        means_tf = self.means_tf
        scales_tf = self.scales_tf
        logits_tf = self.logits_tf
        
        component_distribution_tf = Normal(means_tf, scales_tf)
        component_distribution_tf = Independent(component_distribution_tf, 1)
        
        component_entropy_tf = component_distribution_tf.entropy()
        approx_entropy_tf = tf.reduce_mean(
            tf.reduce_sum(tf.nn.softmax(logits_tf, axis=-1) * component_entropy_tf, axis=-1)
        )
        std_tf = tf.reduce_mean(
            tf.reduce_sum(tf.nn.softmax(logits_tf, axis=-1) * tf.reduce_mean(scales_tf, axis=-1), axis=-1)
        )
        
        mixture_distribution_tf = Categorical(logits=logits_tf)
        dist_tf = MixtureSameFamily(
            mixture_distribution=mixture_distribution_tf,
            component_distribution=component_distribution_tf,
        )

        log_prob_tf = dist_tf.log_prob(self.x_tf)
        
        # Compare results (with tolerance for floating-point differences)
        # Convert TensorFlow tensors to NumPy for comparison
        std_torch_np = std_torch.detach().numpy()
        std_tf_np = std_tf.numpy()
        
        approx_entropy_torch_np = approx_entropy_torch.detach().numpy()
        approx_entropy_tf_np = approx_entropy_tf.numpy()
        
        # Use assertAlmostEqual with a tolerance for floating-point comparisons
        self.assertAlmostEqual(
            float(std_torch_np), 
            float(std_tf_np), 
            places=5, 
            msg="Standard deviations don't match between PyTorch and TensorFlow"
        )
        
        self.assertAlmostEqual(
            float(approx_entropy_torch_np), 
            float(approx_entropy_tf_np), 
            places=5, 
            msg="Approximate entropies don't match between PyTorch and TensorFlow"
        )
        
        # For log_prob, we need to check shape and approximate values
        # This is a more complex tensor, so we'll check if they're close enough
        log_prob_torch_np = log_prob_torch.detach().numpy()
        log_prob_tf_np = log_prob_tf.numpy()
        
        # Check shapes match
        self.assertEqual(
            log_prob_torch_np.shape, 
            log_prob_tf_np.shape, 
            "Log probability shapes don't match"
        )
        
        # Check values are close (using a higher tolerance)
        np.testing.assert_allclose(
            log_prob_torch_np, 
            log_prob_tf_np, 
            rtol=1e-4, 
            atol=1e-4,
            err_msg="Log probabilities don't match between PyTorch and TensorFlow"
        )


if __name__ == '__main__':
    unittest.main()
