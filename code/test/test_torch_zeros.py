import unittest
import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_zeros


class TestTorchZeros(unittest.TestCase):
    """Test case for comparing zeros tensor creation between TensorFlow and PyTorch."""
    
    def test_zeros_direct_shape(self):
        """Test zeros tensor creation with direct shape parameter."""
        shape = (2, 3)
        
        # TensorFlow: zeros tensor
        tf_tensor = torch_zeros(shape, dtype=tf.float32)
        
        # PyTorch: zeros tensor
        torch_tensor = torch.zeros(shape, dtype=torch.float32)
        
        # Check if their outputs are equivalent
        self.assertTrue(np.allclose(tf_tensor.numpy(), torch_tensor.numpy()),
                        "Direct shape: TensorFlow and PyTorch zeros tensors should match")
    
    def test_zeros_unpacked_shape(self):
        """Test zeros tensor creation with unpacked shape parameters (*shape)."""
        shape = (2, 3)
        
        # TensorFlow: zeros tensor
        tf_tensor = torch_zeros(*shape, dtype=tf.float32)
        
        # PyTorch: zeros tensor
        torch_tensor = torch.zeros(*shape, dtype=torch.float32)
        
        # Check if their outputs are equivalent
        self.assertTrue(np.allclose(tf_tensor.numpy(), torch_tensor.numpy()),
                        "Unpacked shape: TensorFlow and PyTorch zeros tensors should match")
    
    def test_zeros_different_shapes(self):
        """Test zeros tensor creation with various shapes."""
        shapes = [(1, 1), (3, 4), (2, 3, 4), (5,)]
        
        for shape in shapes:
            with self.subTest(shape=shape):
                # print("shape = ", shape)
                # Direct shape passing
                tf_tensor = torch_zeros(shape, dtype=tf.float32)
                torch_tensor = torch.zeros(shape, dtype=torch.float32)
                self.assertTrue(np.allclose(tf_tensor.numpy(), torch_tensor.numpy()),
                                f"Shape {shape}: TensorFlow and PyTorch zeros tensors should match")
                
                # Unpacked shape passing
                tf_tensor = torch_zeros(*shape, dtype=tf.float32)
                torch_tensor = torch.zeros(*shape, dtype=torch.float32)
                self.assertTrue(np.allclose(tf_tensor.numpy(), torch_tensor.numpy()),
                                f"Unpacked shape {shape}: TensorFlow and PyTorch zeros tensors should match")


if __name__ == "__main__":
    unittest.main()
