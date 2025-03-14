import tensorflow as tf
import numpy as np
import unittest
import torch

from util.torch_to_tf import torch_nanmean

class TestNanMean(unittest.TestCase):
    def setUp(self):
        # Setup test data
        self.arr = np.array([1.1, 1.2, 1.3, float('nan')])
        self.log_arr = np.array([0.1, 0.2, 0.3, 0.4])
        
    def test_nanmean_tensorflow_vs_pytorch(self):
        # TensorFlow implementation
        ratio = tf.constant(self.arr, dtype=tf.float64)
        logratio = tf.constant(self.log_arr, dtype=tf.float64)
        kl_difference_tf = (ratio - 1) - logratio
        
        print("TensorFlow kl_difference = ", kl_difference_tf)
        
        tf_approx_kl = torch_nanmean(kl_difference_tf)
        print("TensorFlow Approximate KL Divergence:", tf_approx_kl.numpy())
        
        # PyTorch implementation
        ratio_torch = torch.tensor(self.arr)
        logratio_torch = torch.tensor(self.log_arr)
        kl_difference_torch = (ratio_torch - 1) - logratio_torch
        
        print("PyTorch kl_difference = ", kl_difference_torch)
        
        torch_approx_kl = kl_difference_torch.nanmean()
        print("PyTorch Approximate KL Divergence:", torch_approx_kl.numpy())
        
        # Assert that both implementations give the same result
        self.assertTrue(np.allclose(tf_approx_kl.numpy(), torch_approx_kl.numpy()))
    
    def test_nanmean_handles_nan_values(self):
        # Test that nanmean properly ignores NaN values
        tensor_with_nan = tf.constant(self.arr, dtype=tf.float64)
        result = torch_nanmean(tensor_with_nan)
        
        # Calculate expected result manually (average of non-NaN values)
        expected = np.nanmean(self.arr)
        
        self.assertTrue(np.allclose(result.numpy(), expected))

if __name__ == '__main__':
    unittest.main()
