import tensorflow as tf
import torch
import numpy as np

from util.torch_to_tf import torch_nn_init_normal_



# Testing PyTorch and TensorFlow outputs
def test_compare_tf_torch():
    input_data = np.random.rand(1, 5).astype(np.float32)  # Input tensor for both frameworks

    # PyTorch
    torch_layer = torch.nn.Linear(5, 10)
    torch.nn.init.normal_(torch_layer.weight, mean=0.0, std=0.1)  # Initialize weights
    torch_layer.bias.data.fill_(0.0)  # Initialize bias to zero
    torch_input = torch.tensor(input_data)
    torch_output = torch_layer(torch_input).detach().numpy()

    # TensorFlow
    tf_layer = tf.keras.layers.Dense(10, use_bias=True)

    tf_layer.build(input_shape=(None, 5))

    # Copy PyTorch weights and bias to TensorFlow
    torch_weights = torch_layer.weight.detach().numpy().T  # Transpose weights for TensorFlow
    torch_bias = torch_layer.bias.detach().numpy()

    torch_nn_init_normal_(tf_layer.kernel, mean=0.0, std=0.1)  # Use the same initializer
    tf_layer.kernel.assign(torch_weights)
    tf_layer.bias.assign(torch_bias)

    tf_input = tf.convert_to_tensor(input_data)
    tf_output = tf_layer(tf_input).numpy()

    # Compare outputs
    print("PyTorch output:\n", torch_output)
    print("TensorFlow output:\n", tf_output)
    print("Outputs match:", np.allclose(torch_output, tf_output, atol=1e-5))

# if __name__ == "__main__":

test_compare_tf_torch()

