import unittest
import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_multinomial

class TestMultinomial(unittest.TestCase):
    def setUp(self):
        # Create a simple probability distribution with 3 categories
        self.probabilities = np.array([0.2, 0.5, 0.3])  # probability of each category

        self.num_samples = 5

        self.torch_probs = torch.tensor(self.probabilities, dtype=torch.float32)

        self.tf_probs = tf.convert_to_tensor(self.probabilities, dtype=tf.float32)
        # For reproducibility
        # np.random.seed(42)
        # torch.manual_seed(42)
        # tf.random.set_seed(42)

    def test_torch_multinomial(self):
        """Test PyTorch's multinomial sampling function"""
        # Sample using torch.multinomial
        torch_samples = torch.multinomial(self.torch_probs, self.num_samples, replacement=True)

        # Convert to numpy for assertions
        torch_samples_np = torch_samples.numpy()
        
        # Check shape
        self.assertEqual(torch_samples_np.shape[0], self.num_samples, 
                         "Number of samples should match requested amount")
        
        # Check range (should be 0, 1, or 2 since we have 3 categories)
        self.assertTrue(np.all(torch_samples_np >= 0), "Sampled indices should be non-negative")
        self.assertTrue(np.all(torch_samples_np < len(self.probabilities)), 
                        "Sampled indices should be less than number of categories")
        
        print(f"torch.multinomial sampled indices: {torch_samples_np}")

    def test_tensorflow_categorical(self):
        """Test TensorFlow's categorical sampling function"""
        # Convert probabilities to logits for tf.random.categorical
        logits = tf.math.log(self.tf_probs)

        # print("logits[None, :] = ", logits[None, :])
        
        # Sample using tf.random.categorical
        # tf_samples = tf.random.categorical(logits[None, :], self.num_samples, dtype=tf.int32).numpy().flatten()
        tf_samples = torch_multinomial(self.tf_probs, self.num_samples, replacement=True)

        print("tf_samples = ", tf_samples)

        # Check shape
        self.assertEqual(tf_samples.shape[0], self.num_samples, 
                         "Number of samples should match requested amount")
        
        # Check range (should be 0, 1, or 2 since we have 3 categories)
        self.assertTrue(np.all(tf_samples >= 0), "Sampled indices should be non-negative")
        self.assertTrue(np.all(tf_samples < len(self.probabilities)), 
                        "Sampled indices should be less than number of categories")
        
        print(f"tensorflow tf.random.categorical sampled indices: {tf_samples}")


if __name__ == '__main__':
    unittest.main()

