import unittest
from util.torch_to_tf import nn_GroupNorm, torch_tensor_permute
import numpy as np
import torch
import tensorflow as tf


class TestGroupNorm(unittest.TestCase):
    def setUp(self):
        # set random seeds to ensure the reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        tf.random.set_seed(42)

    def test_group_norm_case1(self):
        """Test GroupNorm with batch_size=2, height=4, width=4, channels=8, num_groups=2"""
        # case1:
        batch_size = 2
        height = 4
        width = 4
        channels = 8
        num_groups = 2

        self._run_group_norm_test(batch_size, channels, height, width, num_groups)

    def test_group_norm_case2(self):
        """Test GroupNorm with batch_size=256, channels=64, height=1, width=4, num_groups=8"""
        # case2:
        batch_size = 256
        channels = 64
        height = 1
        width = 4
        num_groups = 8
        # (256, 64, 1, 4)

        self._run_group_norm_test(batch_size, channels, height, width, num_groups)

    def _run_group_norm_test(self, batch_size, channels, height, width, num_groups):
        """Helper method to run GroupNorm test with given parameters"""
        # generate a numpy array
        input_np = np.random.randn(batch_size, channels, height, width).astype(np.float32)

        # convert the numpy array to a PyTorch tensor
        input_torch = torch.from_numpy(input_np)
        # .permute(0, 3, 1, 2)  # (batch_size, height, width, channels) convert to the PyTorch (batch_size, channels, height, width)

        # .permute(0, 2, 3, 1)
        # (batch_size, channels, height, width) convert to (batch_size, height, width, channels)

        # convert the numpy data to the TensorFlow tensor
        # input_tf = torch_tensor_permute(tf.convert_to_tensor(input_np), [0, 2, 3, 1])  # TensorFlow needs (batch_size, height, width, channels)
        input_tf = tf.convert_to_tensor(input_np)  # TensorFlow needs (batch_size, height, width, channels)

        # create PyTorch's GroupNorm layer
        torch_group_norm = torch.nn.GroupNorm(num_groups, channels, eps=1e-5, affine=True)

        # create TensorFlow's GroupNorm layer
        tf_group_norm = nn_GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-5, affine=True)

        # copy PyTorch's gamma and beta parameters to the TensorFlow layer
        with torch.no_grad():
            tf_group_norm.gamma.assign(torch_group_norm.weight.numpy())
            tf_group_norm.beta.assign(torch_group_norm.bias.numpy())

        # forward pass
        # PyTorch
        output_torch = torch_group_norm(input_torch)
        # .permute(0, 2, 3, 1)  # (batch_size, channels, height, width) convert back to (batch_size, height, width, channels)
        output_torch_np = output_torch.detach().numpy()

        # # TensorFlow
        # output_tf_np = torch_tensor_permute(tf_group_norm(input_tf), [0,3,1,2]).numpy()
        output_tf_np = tf_group_norm(input_tf).numpy()

        # # Print outputs for debugging
        # print(f"Testing GroupNorm with batch_size={batch_size}, channels={channels}, height={height}, width={width}, num_groups={num_groups}")
        # print("PyTorch Output shape:", output_torch_np.shape)
        # print("TensorFlow Output shape:", output_tf_np.shape)

        # check if the outputs are consistent
        self.assertTrue(
            np.allclose(output_torch_np, output_tf_np, atol=1e-5),
            "PyTorch and TensorFlow GroupNorm outputs are not consistent"
        )


if __name__ == "__main__":
    unittest.main()
