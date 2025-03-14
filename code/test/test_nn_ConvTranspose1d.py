import unittest
import tensorflow as tf
import torch
import numpy as np

from util.torch_to_tf import nn_ConvTranspose1d


class TestConvTranspose1d(unittest.TestCase):
    def setUp(self):
        # PyTorch ConvTranspose1d
        self.torch_layer = torch.nn.ConvTranspose1d(3, 3, 4, 2, 1)  # padding=1
        self.torch_input = torch.randn(1, 3, 10)
        
        # TensorFlow ConvTranspose1d
        self.tf_layer = nn_ConvTranspose1d(3, 3, 4, 2, 1)  # only implement the padding=1 case
        self.tf_input = tf.convert_to_tensor(self.torch_input.numpy(), dtype=tf.float32)

        tf_output = self.tf_layer(self.tf_input).numpy()
        # Copy weights from PyTorch to TensorFlow
        # same weights
        torch_weights = self.torch_layer.weight.detach().numpy()  # PyTorch weight
        torch_weights_tf = np.transpose(torch_weights, (2, 1, 0))  # convert to the shape of the TensorFlow shape
        self.tf_layer.conv1d_transpose.kernel.assign(torch_weights_tf)
        
        # same bias
        if self.torch_layer.bias is not None:
            self.tf_layer.conv1d_transpose.bias.assign(self.torch_layer.bias.detach().numpy())

    def test_output_shape(self):
        """Test if PyTorch and TensorFlow outputs have the same shape"""
        torch_output = self.torch_layer(self.torch_input).detach().numpy()
        tf_output = self.tf_layer(self.tf_input).numpy()
        
        self.assertEqual(torch_output.shape, tf_output.shape, 
                         f"Shape mismatch: PyTorch {torch_output.shape} vs TensorFlow {tf_output.shape}")
        
    def test_output_values(self):
        """Test if PyTorch and TensorFlow outputs have similar values"""
        torch_output = self.torch_layer(self.torch_input).detach().numpy()
        tf_output = self.tf_layer(self.tf_input).numpy()
        
        max_diff = np.abs(torch_output - tf_output).max()
        self.assertLess(max_diff, 1e-3, f"Max difference too large: {max_diff}")
        
    def test_print_results(self):
        """Print the results for manual inspection (not an actual test)"""
        torch_output = self.torch_layer(self.torch_input).detach().numpy()
        tf_output = self.tf_layer(self.tf_input).numpy()
        
        # print("max difference:", np.abs(torch_output - tf_output).max())
        # print("PyTorch output shape:", torch_output.shape)
        # print("TensorFlow output shape:", tf_output.shape)


if __name__ == '__main__':
    unittest.main()
