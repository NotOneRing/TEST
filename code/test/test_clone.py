import unittest
import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_tensor_clone


class TestClone(unittest.TestCase):
    def test_clone_function(self):
        # Create a test tensor in PyTorch
        torch_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        
        # Clone using PyTorch
        torch_cloned = torch_input.clone()
        
        # Convert PyTorch tensor to TensorFlow tensor
        tf_input = tf.convert_to_tensor(torch_input.detach().numpy())  # Detach to avoid gradient issues
        tf_cloned = torch_tensor_clone(tf_input)

        # Convert TensorFlow result back to NumPy for comparison
        tf_cloned_np = tf_cloned.numpy()
        torch_cloned_np = torch_cloned.detach().numpy()
        
        # Compare values
        np.testing.assert_allclose(torch_cloned_np, tf_cloned_np, rtol=1e-6, atol=1e-6)
        
        # Check independence
        torch_cloned[0, 0] = 10.0
        tf_cloned = tf_cloned + 1  # Modify the cloned TensorFlow tensor
        
        self.assertNotEqual(torch_input[0, 0].item(), torch_cloned[0, 0].item(), 
                           "PyTorch clone is not independent!")
        self.assertFalse(np.allclose(tf_input.numpy(), tf_cloned.numpy()), 
                        "TensorFlow clone is not independent!")


if __name__ == '__main__':
    unittest.main()
