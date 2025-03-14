import torch
import tensorflow as tf
import numpy as np
import unittest

# Import the torch_rand function from util.torch_to_tf
from util.torch_to_tf import torch_rand

class TestRand(unittest.TestCase):
    """
    Test class for comparing PyTorch's rand function with its TensorFlow implementation.
    """
    
    def test_rand_unpacked_shape(self):
        """
        Test torch.rand and torch_rand with unpacked shape (*shape).
        """
        # Define shape for random tensor
        shape = (3, 4)

        # PyTorch: torch.rand
        pytorch_tensor = torch.rand(*shape)
        
        # TensorFlow: torch_rand (wrapper for tf.random.uniform)
        tensorflow_tensor = torch_rand(*shape)
        
        # Check if the shapes are equal
        self.assertEqual(pytorch_tensor.shape, tensorflow_tensor.shape)
        
        # # Print tensors for debugging (optional)
        # print("PyTorch Tensor (torch.rand):")
        # print(pytorch_tensor)
        # print("\nTensorFlow Tensor (torch_rand wrapper):")
        # print(tensorflow_tensor)
    
    def test_rand_packed_shape(self):
        """
        Test torch.rand and torch_rand with packed shape (shape).
        """
        # Define shape for random tensor
        shape = (3, 4)

        # PyTorch: torch.rand
        pytorch_tensor = torch.rand(shape)
        
        # TensorFlow: torch_rand (wrapper for tf.random.uniform)
        tensorflow_tensor = torch_rand(shape)
        
        # Check if the shapes are equal
        self.assertEqual(pytorch_tensor.shape, tensorflow_tensor.shape)
        
        # # Print tensors for debugging (optional)
        # print("PyTorch Tensor (torch.rand):")
        # print(pytorch_tensor)
        # print("\nTensorFlow Tensor (torch_rand wrapper):")
        # print(tensorflow_tensor)

if __name__ == '__main__':
    unittest.main()
