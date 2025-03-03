import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_nn_init_ones_, nn_Linear


# Define a test for both PyTorch and TensorFlow
def test_ones_initialization():
    # PyTorch implementation
    torch_layer = torch.nn.Linear(5, 10)
    torch.nn.init.ones_(torch_layer.weight)
    torch.nn.init.ones_(torch_layer.bias)

    input_data = np.random.rand(1, 5).astype(np.float32)

    # TensorFlow implementation
    tf_layer = nn_Linear(5, 10)
    # tf_layer.build(input_shape=(None, 5))  # Build the layer to initialize variables
    tf_input = tf.constant(input_data)
    tf_output = tf_layer(tf_input).numpy()

    torch_nn_init_ones_(tf_layer.kernel)
    torch_nn_init_ones_(tf_layer.bias)

    # Create identical input data

    # PyTorch forward pass
    torch_input = torch.tensor(input_data)
    torch_output = torch_layer(torch_input).detach().numpy()

    # TensorFlow forward pass
    # tf_input = tf.constant(input_data)
    tf_output = tf_layer(tf_input).numpy()

    # Compare outputs
    print("Torch output:\n", torch_output)
    print("TensorFlow output:\n", tf_output)
    print("Outputs match:", np.allclose(torch_output, tf_output, atol=1e-5))

# Run the test
# if __name__ == "__main__":
test_ones_initialization()
