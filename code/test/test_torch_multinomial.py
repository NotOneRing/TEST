import unittest
import torch
import tensorflow as tf

from util.torch_to_tf import torch_multinomial

class TestMultinomial(unittest.TestCase):
    def test_torch_multinomial(self):
        """Test PyTorch multinomial sampling functionality.
        
        Creates a weights tensor and samples from the distribution with replacement.
        """
        # weights tensor
        weights = torch.tensor([0.1, 0.3, 0.4, 0.2])

        # take out 5 samples from the distribution(sampling with replacement)
        samples = torch.multinomial(weights, 5, replacement=True)
        
        # Verify samples shape and type
        self.assertEqual(samples.shape, torch.Size([5]))
        self.assertTrue(all(0 <= idx < 4 for idx in samples))

    def test_tensorflow_categorical(self):
        """Test TensorFlow categorical sampling functionality.
        
        Sets a random seed, creates a logits tensor, and samples from the distribution
        with replacement.
        """
        # fix seed
        tf.random.set_seed(42)
        
        # logits tensor
        logits = tf.constant([0.1, 0.3, 0.4, 0.2])

        # print("logits.shape = ", logits.shape)

        # take out 5 samples from the distribution(sampling with replacement)
        # samples = tf.random.categorical(logits, num_samples=5)
        samples = torch_multinomial(logits, 5, replacement=True)
        
        # print("samples = ", samples)
        # print("samples.shape = ", samples.shape)

        # Verify samples shape and type
        self.assertEqual(samples.shape, (5,))
        self.assertTrue(all(0 <= idx < 4 for idx in samples))


if __name__ == '__main__':
    unittest.main()
