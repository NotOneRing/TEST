import unittest
from util.torch_to_tf import torch_tensor, torch_zeros_like

import numpy as np
import tensorflow as tf
import torch


class TestZerosLike(unittest.TestCase):
    """Test case for zeros_like function comparison between PyTorch and TensorFlow."""
    
    def test_zeros_like(self):
        """Test that torch_zeros_like produces the same result as torch.zeros_like."""
        np_arr = np.array(range(9)).reshape(3, 3).astype(np.float32)

        # Convert numpy array to float32 type
        
        # Create PyTorch tensor from numpy array
        tensor_torch = torch.tensor(np_arr)
        
        # Create TensorFlow tensor from numpy array
        tensor_tf = torch_tensor(np_arr)
        
        # Apply zeros_like operation on TensorFlow tensor
        tf_tensor_zeros_like = torch_zeros_like(tensor_tf)
        
        # # Print TensorFlow result for debugging
        # print("tf_tensor_zeros_like = ", tf_tensor_zeros_like)
        
        # Apply zeros_like operation on PyTorch tensor
        torch_tensor_zeros_like = torch.zeros_like(tensor_torch)
        
        # # Print PyTorch result for debugging
        # print("torch_tensor_zeros_like = ", torch_tensor_zeros_like)
        
        # Assert that both operations produce the same result
        self.assertTrue(np.allclose(tf_tensor_zeros_like.numpy(), torch_tensor_zeros_like.numpy()))


if __name__ == '__main__':
    unittest.main()
