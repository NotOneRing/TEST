import unittest
import torch
import tensorflow as tf
import numpy as np
from torch.distributions.categorical import Categorical as TorchCategorical

from util.torch_to_tf import Categorical as CategoricalDistribution



class TestCategoricalDistribution(unittest.TestCase):
    def setUp(self):
        # Common test data
        # self.probs = torch.tensor([0.1, 0.7, 0.2])
        # self.logits = torch.tensor([0.5, 1.0, -0.5])
        # self.logits_batch = torch.tensor([[1.0, 0.5, -0.5], [0.2, 0.7, 0.1]])

        self.probs = np.array([0.1, 0.7, 0.2])
        self.logits = np.array([0.5, 1.0, -0.5])
        self.logits_batch = np.array([[1.0, 0.5, -0.5], [0.2, 0.7, 0.1]])


    def test_initialization(self):
        # Test initialization with probs
        dist_from_probs = CategoricalDistribution( probs= tf.convert_to_tensor(self.probs) )
        self.assertTrue(np.allclose( dist_from_probs.probs.numpy(), self.probs ))
        
        # Test initialization with logits
        dist_from_logits = CategoricalDistribution(logits= tf.convert_to_tensor(self.logits) )
        expected_probs = tf.nn.softmax( tf.convert_to_tensor(self.logits), axis=-1)

        # print("expected_probs = ", expected_probs)

        self.assertTrue(np.allclose(dist_from_logits.probs.numpy(), expected_probs.numpy()))
        
        # Test initialization with neither probs nor logits
        with self.assertRaises(ValueError):
            CategoricalDistribution()
    


    def test_sample(self):
        # Test that sample returns a tensor with the expected shape
        dist = CategoricalDistribution(probs= tf.convert_to_tensor(self.probs) )
        sample = dist.sample()

        # print("sample = ", sample)
        # print("sample.shape = ", sample.shape)

        self.assertEqual(sample.shape, tf.TensorShape([]))
        self.assertTrue(0 <= sample.numpy() < len(self.probs))
        
        # Test with batch
        dist_batch = CategoricalDistribution(logits= tf.convert_to_tensor(self.logits_batch) )
        samples_batch = dist_batch.sample()

        # print("samples_batch = ", samples_batch)
        # print("samples_batch.shape = ", samples_batch.shape)

        self.assertEqual(samples_batch.shape, tf.TensorShape([2]))
        for i in range(samples_batch.shape[0]):
            # print("type(samples_batch[i]) = ", type(samples_batch[i]))
            self.assertTrue(0 <= samples_batch[i].numpy() < self.logits_batch.shape[1])
    


    def test_log_prob(self):
        # Test log_prob with a single value
        dist = CategoricalDistribution(probs= tf.convert_to_tensor(self.probs) )
        x = tf.convert_to_tensor([[1]])  # Class 1 (index 1)
        log_prob = dist.log_prob(x)

        # expected_log_prob = tf.math.log(tf.convert_to_tensor([0.1, 0.7, 0.2]))  # Log of probability for class 1
        # print("expected_log_prob = ", expected_log_prob)

        expected_log_prob = tf.math.log(tf.convert_to_tensor([0.7]))  # Log of probability for class 1

        # print("log_prob = ", log_prob)
        # print("expected_log_prob = ", expected_log_prob)
        self.assertTrue(np.allclose(log_prob.numpy(), expected_log_prob.numpy()))
    
    def test_entropy(self):
        # Test entropy calculation
        dist = CategoricalDistribution(probs= tf.convert_to_tensor(self.probs) )
        entropy = dist.entropy()
        expected_entropy = - tf.reduce_sum( tf.convert_to_tensor(self.probs) * tf.math.log(self.probs) )
        self.assertTrue(np.allclose(entropy.numpy(), expected_entropy.numpy()))
    
    def test_comparison_with_torch_categorical(self):
        # Compare with PyTorch's built-in Categorical
        # For single distribution
        custom_dist = CategoricalDistribution(probs= tf.convert_to_tensor(self.probs) )
        torch_dist = TorchCategorical(probs= torch.tensor(self.probs) )
        
        # Entropy should be the same
        custom_entropy = custom_dist.entropy()
        torch_entropy = torch_dist.entropy()
        self.assertTrue(np.allclose(custom_entropy.numpy(), torch_entropy.numpy()))
        
        # For batch distribution
        custom_dist_batch = CategoricalDistribution(logits= tf.convert_to_tensor(self.logits_batch) )
        torch_dist_batch = TorchCategorical(logits= torch.tensor(self.logits_batch) )
        
        # Entropy should be the same
        custom_entropy_batch = custom_dist_batch.entropy()
        torch_entropy_batch = torch_dist_batch.entropy()
        self.assertTrue(np.allclose(custom_entropy_batch.numpy(), torch_entropy_batch.numpy()))


if __name__ == '__main__':
    unittest.main()



