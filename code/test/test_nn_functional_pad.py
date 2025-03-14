import unittest
import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
from util.torch_to_tf import torch_randn, nn_functional_pad


class TestNNFunctionalPad(unittest.TestCase):
    """Test case for nn.functional.pad implementation."""
    
    def test_pad_replicate(self):
        """Test padding with replicate mode."""
        # Create a random tensor
        x = torch_randn(2, 3, 4)
        
        # Convert to PyTorch tensor
        x_torch = torch.tensor(x.numpy())
        
        # Padding value
        pad_value = 2
        
        # Apply padding using PyTorch
        x_torch_padded_torch = F.pad(x_torch, (pad_value, pad_value, pad_value, pad_value), mode='replicate')
        
        # print("\nPyTorch nn.pad replicate padded tensor:")
        # print(x_torch_padded_torch)
        
        # Apply padding using our implementation
        result = nn_functional_pad(x, (pad_value, pad_value, pad_value, pad_value), mode="replicate")
        
        # print("result.shape = ", result.shape)
        
        # Compare results
        self.assertTrue(np.allclose(result.numpy(), x_torch_padded_torch.numpy()))
        
        print("np.allclose(result.numpy(), x_torch_padded_torch.numpy()) = ", 
              np.allclose(result.numpy(), x_torch_padded_torch.numpy()))


if __name__ == '__main__':
    unittest.main()
