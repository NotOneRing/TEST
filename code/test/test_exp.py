import unittest
import torch
import tensorflow as tf
import numpy as np
from util.torch_to_tf import torch_exp

class TestExp(unittest.TestCase):
    def test_exp(self):
        # Generate random input data
        input_data = np.random.randn(3, 4).astype(np.float32)
        
        # PyTorch implementation
        input_tensor_torch = torch.tensor(input_data)
        output_torch = torch.exp(input_tensor_torch)
        
        # TensorFlow implementation
        input_tensor_tf = tf.convert_to_tensor(input_data)
        output_tf = torch_exp(input_tensor_tf)
        
        # # Print results for debugging
        # print("Input data:\n", input_data)
        # print("\nPyTorch exp output:\n", output_torch.numpy())
        # print("\nTensorFlow exp output:\n", output_tf.numpy())
        
        # Calculate and print difference
        difference = np.abs(output_torch.numpy() - output_tf.numpy())
        # print("\nDifference between PyTorch and TensorFlow outputs:\n", difference)
        
        # Assert that outputs are close enough
        self.assertTrue(np.allclose(output_torch.numpy(), output_tf.numpy()))

if __name__ == '__main__':
    unittest.main()


