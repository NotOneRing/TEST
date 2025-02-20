from util.torch_to_tf import nn_Linear

import tensorflow as tf
import torch
import numpy as np

def test_nn_Linear():

    # PyTorch Implementation of nn.Linear
    torch_linear = torch.nn.Linear(in_features=4, out_features=3, bias=True)
    tf_linear = nn_Linear(in_features=4, out_features=3)

    # # Copy PyTorch parameters to TensorFlow model
    # tf_linear.model.kernel.assign(torch_linear.weight.detach().numpy().T)
    # tf_linear.model.bias.assign(torch_linear.bias.detach().numpy())

    _ = tf_linear(tf.constant(np.random.randn(1, 4).astype(np.float32)))

    # Initialize TensorFlow linear layer weights with PyTorch linear layer weights
    tf_linear.model.trainable_weights[0].assign(torch_linear.weight.detach().numpy().T)  # kernel
    tf_linear.model.trainable_weights[1].assign(torch_linear.bias.detach().numpy())     # bias

    # Test input
    np.random.seed(42)
    input_data = np.random.randn(2, 4).astype(np.float32)  # Batch size 2, input features 4

    # Forward pass in PyTorch
    torch_input = torch.tensor(input_data, dtype=torch.float32)
    torch_output = torch_linear(torch_input).detach().numpy()

    # Forward pass in TensorFlow
    tf_input = tf.convert_to_tensor(input_data, dtype=tf.float32)
    tf_output = tf_linear(tf_input).numpy()

    # Comparison
    print("Input Data:\n", input_data)
    print("\nPyTorch Output:\n", torch_output)
    print("\nTensorFlow Output:\n", tf_output)
    print("\nOutputs close:", np.allclose(torch_output, tf_output))
    assert np.allclose(torch_output, tf_output)
















test_nn_Linear()

