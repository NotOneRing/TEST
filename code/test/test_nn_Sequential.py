import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import unittest

from util.torch_to_tf import nn_Linear, nn_Sequential, nn_ReLU


class TestNNSequential(unittest.TestCase):
    def test_nn_Sequential(self):
        # PyTorch test
        pytorch_model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        # Apply Xavier initialization directly to layers
        torch.nn.init.xavier_uniform_(pytorch_model[0].weight)  # For the first linear layer
        torch.nn.init.xavier_uniform_(pytorch_model[2].weight)  # For the second linear layer
        torch.nn.init.zeros_(pytorch_model[0].bias)            # Zero bias for the first linear layer
        torch.nn.init.zeros_(pytorch_model[2].bias)            # Zero bias for the second linear layer

        pytorch_input = torch.randn(5, 4)  # Input tensor with shape (5, 4)
        pytorch_output = pytorch_model(pytorch_input).detach().numpy()
        print("PyTorch output:")
        print(pytorch_output)

        tf_model = nn_Sequential(
            nn_Linear(4, 8),
            nn_ReLU(),
            nn_Linear(8, 2),
        )

        pytorch_weight1 = pytorch_model[0].weight.detach().numpy()
        pytorch_bias1 = pytorch_model[0].bias.detach().numpy()
        pytorch_weight2 = pytorch_model[2].weight.detach().numpy()
        pytorch_bias2 = pytorch_model[2].bias.detach().numpy()

        # # Manually set the weights in TensorFlow to match the initialization in PyTorch
        # tf_model[0].kernel_initializer = tf.keras.initializers.GlorotUniform()
        # tf_model[0].bias_initializer = tf.keras.initializers.Zeros()
        # tf_model[2].kernel_initializer = tf.keras.initializers.GlorotUniform()
        # tf_model[2].bias_initializer = tf.keras.initializers.Zeros()

        # tf_model.build((None, 4))  # This ensures that the model is initialized correctly

        _ = tf_model(tf.constant(np.random.randn(1, 4).astype(np.float32)))

        # Manually set the weights and biases in TensorFlow to match PyTorch
        # tf_model[0].set_weights([pytorch_weight1.T, pytorch_bias1])  # PyTorch weight is (8, 4), TensorFlow expects (4, 8)
        # tf_model[2].set_weights([pytorch_weight2.T, pytorch_bias2])  # Same for the second layer

        tf_model[0].trainable_weights[0].assign(pytorch_weight1.T)  # kernel
        tf_model[0].trainable_weights[1].assign(pytorch_bias1)     # bias

        tf_model[2].trainable_weights[0].assign(pytorch_weight2.T)  # kernel
        tf_model[2].trainable_weights[1].assign(pytorch_bias2)     # bias

        print("len(tf_model) = ", len(tf_model))
        print("len(pytorch_model) = ", len(pytorch_model))

        tf_input = tf.convert_to_tensor(pytorch_input.numpy())  # Convert PyTorch input to TensorFlow tensor
        tf_output = tf_model(tf_input).numpy()
        print("\nTensorFlow output:")
        print(tf_output)

        # Compare results
        if np.allclose(pytorch_output, tf_output, atol=1e-3):
            print("\nThe results match!")
        else:
            print("\nThe results do not match!")

        # Use unittest assertion instead of assert
        self.assertTrue(np.allclose(pytorch_output, tf_output, atol=1e-3), 
                        "PyTorch and TensorFlow outputs do not match")


if __name__ == "__main__":
    unittest.main()
