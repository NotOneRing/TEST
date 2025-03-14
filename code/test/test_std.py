import torch
import tensorflow as tf
import numpy as np
import unittest

from util.torch_to_tf import torch_std


class TestStd(unittest.TestCase):
    """
    Test class for comparing torch.std and its TensorFlow implementation.
    """
    
    def setUp(self):
        """Set up test data that will be used across all test methods."""
        # Generate random test data
        np.random.seed(42)  # For reproducibility
        self.data = np.random.rand(3, 4).astype(np.float32)
        
    def test_std_dim1_correction1_keepdim(self):
        """Test std with dim=1, correction=1, keepdim=True."""
        # Calculate std in PyTorch
        pytorch_tensor = torch.tensor(self.data)
        pytorch_result = torch.std(pytorch_tensor, dim=1, correction=1, keepdim=True)
        
        # Calculate std in TensorFlow
        tensorflow_tensor = tf.convert_to_tensor(self.data)
        tensorflow_result = torch_std(tensorflow_tensor, dim=1, correction=1, keepdim=True)
        
        # Compare results
        self.assertTrue(np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6),
                        "Results don't match for dim=1, correction=1, keepdim=True")
    
    def test_std_dim1_correction0_keepdim(self):
        """Test std with dim=1, correction=0, keepdim=True."""
        # Calculate std in PyTorch
        pytorch_tensor = torch.tensor(self.data)
        pytorch_result = torch.std(pytorch_tensor, dim=1, correction=0, keepdim=True)
        
        # Calculate std in TensorFlow
        tensorflow_tensor = tf.convert_to_tensor(self.data)
        tensorflow_result = torch_std(tensorflow_tensor, dim=1, correction=0, keepdim=True)
        
        # Compare results
        self.assertTrue(np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6),
                        "Results don't match for dim=1, correction=0, keepdim=True")
    
    def test_std_dim1_correction0_5_keepdim(self):
        """Test std with dim=1, correction=0.5, keepdim=True."""
        # Calculate std in PyTorch
        pytorch_tensor = torch.tensor(self.data)
        pytorch_result = torch.std(pytorch_tensor, dim=1, correction=0.5, keepdim=True)
        
        # Calculate std in TensorFlow
        tensorflow_tensor = tf.convert_to_tensor(self.data)
        tensorflow_result = torch_std(tensorflow_tensor, dim=1, correction=0.5, keepdim=True)
        
        # Compare results
        self.assertTrue(np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6),
                        "Results don't match for dim=1, correction=0.5, keepdim=True")
    
    def test_std_no_dim_correction0_5_keepdim(self):
        """Test std with no dim specified, correction=0.5, keepdim=True."""
        # Calculate std in PyTorch
        pytorch_tensor = torch.tensor(self.data)
        pytorch_result = torch.std(pytorch_tensor, correction=0.5, keepdim=True)
        
        # Calculate std in TensorFlow
        tensorflow_tensor = tf.convert_to_tensor(self.data)
        tensorflow_result = torch_std(tensorflow_tensor, correction=0.5, keepdim=True)
        
        # Compare results
        self.assertTrue(np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6),
                        "Results don't match for no dim, correction=0.5, keepdim=True")
    
    def test_std_no_dim_correction0_5_no_keepdim(self):
        """Test std with no dim specified, correction=0.5, no keepdim specified."""
        # Calculate std in PyTorch
        pytorch_tensor = torch.tensor(self.data)
        pytorch_result = torch.std(pytorch_tensor, correction=0.5)
        
        # Calculate std in TensorFlow
        tensorflow_tensor = tf.convert_to_tensor(self.data)
        tensorflow_result = torch_std(tensorflow_tensor, correction=0.5)
        
        # Compare results
        self.assertTrue(np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6),
                        "Results don't match for no dim, correction=0.5, no keepdim")
    

if __name__ == "__main__":
    unittest.main()
