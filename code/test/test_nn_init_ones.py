import unittest
import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_nn_init_ones_, nn_Linear


class TestNNInitOnes(unittest.TestCase):
    """
    Test case for comparing PyTorch and TensorFlow implementations of
    ones initialization for neural network layers.
    """

    def test_ones_initialization(self):
        """
        Test that verifies the equivalence between PyTorch and TensorFlow
        implementations of ones initialization for linear layers.
        """
        # PyTorch implementation
        torch_layer = torch.nn.Linear(5, 10)
        torch.nn.init.ones_(torch_layer.weight)
        torch.nn.init.ones_(torch_layer.bias)

        # Create input data
        input_data = np.random.rand(1, 5).astype(np.float32)

        # TensorFlow implementation
        tf_layer = nn_Linear(5, 10)
        tf_input = tf.constant(input_data)
        tf_output = tf_layer(tf_input).numpy()  # Initialize model weights

        # Initialize TensorFlow weights with ones
        torch_nn_init_ones_(tf_layer.kernel)
        torch_nn_init_ones_(tf_layer.bias)

        # PyTorch forward pass
        torch_input = torch.tensor(input_data)
        torch_output = torch_layer(torch_input).detach().numpy()

        # TensorFlow forward pass after initialization
        tf_output = tf_layer(tf_input).numpy()

        # Compare outputs with assertion instead of print
        self.assertTrue(
            np.allclose(torch_output, tf_output, atol=1e-5),
            f"Outputs do not match:\nTorch output:\n{torch_output}\nTensorFlow output:\n{tf_output}"
        )


if __name__ == "__main__":
    unittest.main()
