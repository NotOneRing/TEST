import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import unittest

from util.torch_to_tf import nn_Conv2d


class TestConv2d(unittest.TestCase):
    
    def test_conv2d_with_bias(self):
        """Test Conv2d with bias=True"""
        # PyTorch Conv2d
        torch_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, bias=True)
        torch_conv.weight.data = torch.nn.init.kaiming_uniform_(torch_conv.weight, nonlinearity='relu')

        # TensorFlow Conv2d
        tf_conv = nn_Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, bias=True)
        
        # Initialize with a sample input
        tf_input = tf.random.uniform([1, 3, 64, 64])
        _ = tf_conv(tf_input).numpy()

        # Set the same weights
        torch_weights = torch_conv.weight.detach().numpy()  # PyTorch weights
        # torch (out_channel, in_channel // groups, kernel_size[0], kernel_size[1])
        # tensorflow (kernel_size[0], kernel_size[1], in_channel // groups, out_channel)
        torch_weights_tf = np.transpose(torch_weights, (2, 3, 1, 0))  # convert to the shape of the TensorFlow shape
        tf_conv.conv2d.kernel.assign(torch_weights_tf)

        # Set the same bias
        if torch_conv.bias is not None:
            tf_conv.conv2d.bias.assign(torch_conv.bias.detach().numpy())

        # Generate the same input
        torch_input = torch.randn(1, 3, 64, 64)  # PyTorch input
        tf_input = tf.convert_to_tensor(torch_input.numpy(), dtype=tf.float32)  # convert to the TensorFlow format

        # Calculate the output
        torch_output = torch_conv(torch_input).detach().numpy()
        tf_output = tf_conv(tf_input).numpy()

        # Calculate the difference
        diff = np.abs(torch_output - tf_output).max()

        # print("torch_output = ", torch_output)
        # print("tf_output = ", tf_output)
        # print("diff = ", diff)

        # Assert outputs are close
        self.assertTrue(np.allclose(torch_output, tf_output, atol=1e-2))

    def test_conv2d_without_bias(self):
        """Test Conv2d with bias=False"""
        # PyTorch Conv2d
        torch_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, bias=False)
        torch_conv.weight.data = torch.nn.init.kaiming_uniform_(torch_conv.weight, nonlinearity='relu')

        # TensorFlow Conv2d
        tf_conv = nn_Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, bias=False)
        
        # Initialize with a sample input
        tf_input = tf.random.uniform([1, 3, 64, 64])
        _ = tf_conv(tf_input).numpy()

        # Set the same weights
        torch_weights = torch_conv.weight.detach().numpy()  # PyTorch weights
        torch_weights_tf = np.transpose(torch_weights, (2, 3, 1, 0))  # convert to the TensorFlow shape
        tf_conv.conv2d.kernel.assign(torch_weights_tf)

        # Generate the same input
        torch_input = torch.randn(1, 3, 64, 64)  # PyTorch input
        tf_input = tf.convert_to_tensor(torch_input.numpy(), dtype=tf.float32)  # convert to the TensorFlow format

        # Calculate the outputs
        torch_output = torch_conv(torch_input).detach().numpy()
        tf_output = tf_conv(tf_input).numpy()

        # Calculate the difference
        diff = np.abs(torch_output - tf_output).max()

        # print("torch_output = ", torch_output)
        # print("tf_output = ", tf_output)
        # print("diff = ", diff)
        
        # Assert outputs are close
        self.assertTrue(np.allclose(torch_output, tf_output, atol=1e-2))


if __name__ == '__main__':
    unittest.main()
