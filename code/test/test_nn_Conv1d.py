import unittest
import numpy as np
import torch
import tensorflow as tf
from util.torch_to_tf import nn_Conv1d


class TestNNConv1d(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        tf.random.set_seed(42)

    def test_conv1d_case1(self):
        # Case 1 parameters
        batch_size = 2
        in_channels = 4
        out_channels = 8
        length = 10
        kernel_size = 3
        stride = 1
        padding = 1
        dilation = 1
        groups = 1  # ensure in_channels % groups == 0

        # Run the test with these parameters
        self._run_conv1d_test(
            batch_size, in_channels, out_channels, length,
            kernel_size, stride, padding, dilation, groups
        )

    def test_conv1d_case2(self):
        # Case 2 parameters
        batch_size = 256
        in_channels = 7
        out_channels = 64
        length = 4
        kernel_size = 1
        stride = 1
        padding = 0
        dilation = 1
        groups = 1

        # Run the test with these parameters
        self._run_conv1d_test(
            batch_size, in_channels, out_channels, length,
            kernel_size, stride, padding, dilation, groups
        )

    def _run_conv1d_test(self, batch_size, in_channels, out_channels, length,
                         kernel_size, stride, padding, dilation, groups):
        # Generate the input data (PyTorch format: batch_size, in_channels, length)
        input_np = np.random.randn(batch_size, in_channels, length).astype(np.float32)
        input_torch = torch.from_numpy(input_np)
        input_tf = tf.convert_to_tensor(input_np)

        # Create a PyTorch layer
        torch_conv1d = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True
        )

        # Forward pass through PyTorch layer
        output_torch = torch_conv1d(input_torch).detach().numpy()

        # Create a TensorFlow layer
        tf_conv1d = nn_Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True
        )

        output_tf = tf_conv1d(input_tf).numpy()  

        # Copy weights and bias from PyTorch to TensorFlow
        with torch.no_grad():
            # Adjust weights' shape
            # PyTorch: (out_channels, in_channels, kernel_size)
            # TensorFlow: (kernel_size, in_channels, out_channels)
            torch_weight = torch_conv1d.weight.numpy().transpose(2, 1, 0)
            torch_bias = torch_conv1d.bias.numpy()
            
            # Check if the weights of the TF layer are already initialized
            if len(tf_conv1d.conv1d.weights) == 0:
                raise RuntimeError("Conv1D not initialized, please check the version of the TensorFlow or the shape of the input")
            
            # Assign weights and biases
            tf_conv1d.conv1d.kernel.assign(torch_weight)
            tf_conv1d.conv1d.bias.assign(torch_bias)

        # Forward pass through TensorFlow layer
        output_tf = tf_conv1d(input_tf).numpy()

        # print("output_torch = ", output_torch)
        # print("output_tf = ", output_tf)

        # Check consistency
        self.assertTrue(
            np.allclose(output_torch, output_tf, atol=1e-3),
            f"Outputs are NOT consistent for parameters: batch_size={batch_size}, "
            f"in_channels={in_channels}, out_channels={out_channels}, length={length}, "
            f"kernel_size={kernel_size}, stride={stride}, padding={padding}, "
            f"dilation={dilation}, groups={groups}"
        )


if __name__ == '__main__':
    unittest.main()
