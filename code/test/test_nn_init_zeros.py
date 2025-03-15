import unittest
import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_nn_init_zeros_, nn_Linear


class TestNNInitZeros(unittest.TestCase):
    def test_zeros_initialization(self):
        """
        Test that zeros initialization works identically in PyTorch and TensorFlow.
        """
        # PyTorch implementation
        torch_layer = torch.nn.Linear(5, 10)
        torch.nn.init.zeros_(torch_layer.weight)
        torch.nn.init.zeros_(torch_layer.bias)

        # TensorFlow implementation
        tf_layer = nn_Linear(5, 10)
        # tf_layer.build(input_shape=(None, 5))  # Build the layer to initialize variables
        # torch_nn_init_zeros_(tf_layer.kernel)
        # torch_nn_init_zeros_(tf_layer.bias)
        # print("tf_layer.kernel = ", tf_layer.kernel)
        # print("tf_layer.bias = ", tf_layer.bias)

        # Create identical input data
        input_data = np.random.rand(1, 5).astype(np.float32)

        # PyTorch forward pass
        torch_input = torch.tensor(input_data)
        torch_output = torch_layer(torch_input).detach().numpy()

        # TensorFlow forward pass
        tf_input = tf.constant(input_data)

        # Only build() is not used. Run the program to initialize
        tf_output = tf_layer(tf_input).numpy()

        # Initialize TensorFlow weights and biases to zeros
        torch_nn_init_zeros_(tf_layer.kernel)
        torch_nn_init_zeros_(tf_layer.bias)
        
        # Run forward pass with initialized weights
        tf_output = tf_layer(tf_input).numpy()

        # Compare outputs with assertion instead of print
        self.assertTrue(
            np.allclose(torch_output, tf_output, atol=1e-5),
            f"Outputs do not match:\nTorch output:\n{torch_output}\nTensorFlow output:\n{tf_output}"
        )


if __name__ == "__main__":
    unittest.main()
