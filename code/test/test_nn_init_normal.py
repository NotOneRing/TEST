import tensorflow as tf
import torch
import numpy as np
import unittest

from util.torch_to_tf import torch_nn_init_normal_, torch_nn_init_zeros_, nn_Linear


class TestNNInitNormal(unittest.TestCase):
    """
    Test case for comparing PyTorch and TensorFlow normal initialization
    """
    
    def test_compare_tf_torch(self):
        """
        Testing PyTorch and TensorFlow outputs with normal initialization
        """
        input_data = np.random.rand(1, 5).astype(np.float32)  # Input tensor for both frameworks

        # PyTorch
        torch_layer = torch.nn.Linear(5, 10)
        torch.nn.init.normal_(torch_layer.weight, mean=0.0, std=0.1)  # Initialize weights
        torch_layer.bias.data.fill_(0.0)  # Initialize bias to zero
        torch_input = torch.tensor(input_data)
        torch_output = torch_layer(torch_input).detach().numpy()

        # print("torch_layer.weight = ", torch_layer.weight)

        # TensorFlow
        # tf_layer = tf.keras.layers.Dense(10, use_bias=True)
        tf_layer = nn_Linear(5, 10)

        tf_layer.build(input_shape=(None, 5))

        # Copy PyTorch weights and bias to TensorFlow
        torch_weights = torch_layer.weight.detach().numpy().T  # Transpose weights for TensorFlow
        torch_bias = torch_layer.bias.detach().numpy()

        tf_input = tf.convert_to_tensor(input_data)

        tf_output = tf_layer(tf_input).numpy()

        # print("type(tf_layer.kernel) = ", type(tf_layer.kernel))
        # print("isinstance(tf_layer.kernel, tf.Variable) = ", isinstance(tf_layer.kernel, tf.Variable))
        
        torch_nn_init_normal_(tf_layer.kernel, mean=0.0, std=0.1)  # Use the same initializer
        torch_nn_init_zeros_(tf_layer.bias)  # Use the same initializer
        # tf_layer.kernel.assign(torch_weights)
        # tf_layer.bias.assign(torch_bias)

        tf_output = tf_layer(tf_input).numpy()

        # print("tf_layer.kernel = ", tf_layer.kernel)
        # print("tf_layer.bias = ", tf_layer.bias)

        # # Compare outputs
        # print("PyTorch output:\n", torch_output)
        # print("TensorFlow output:\n", tf_output)

        # Add assertions to verify the results
        self.assertTrue(np.allclose(torch_layer.bias.data.numpy(), tf_layer.bias.numpy(), atol=1e-5), 
                        "Bias values do not match between PyTorch and TensorFlow")
        
        # We don't check exact equality of outputs since normal initialization is random,
        # but we can check that the shapes match
        self.assertEqual(torch_output.shape, tf_output.shape, 
                         "Output shapes do not match between PyTorch and TensorFlow")


if __name__ == "__main__":
    unittest.main()
