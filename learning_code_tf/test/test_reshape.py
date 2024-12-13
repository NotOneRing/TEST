
from util.torch_to_tf import torch_reshape

import torch
import tensorflow as tf
import numpy as np

# def torch_reshape(input, shape):
#     return tf.reshape(input, shape)

# Generate test cases for torch.reshape and torch_reshape
def test_reshape():
    # Create a random tensor with a fixed shape
    input_tensor_torch = torch.randn(2, 3, 4)
    input_tensor_tf = tf.convert_to_tensor(input_tensor_torch.numpy())

    # Define target shape
    target_shape = (4, 6)

    # Perform reshape in PyTorch
    torch_result = torch.reshape(input_tensor_torch, target_shape)

    # Perform reshape in TensorFlow
    tf_result = torch_reshape(input_tensor_tf, *target_shape)

    # tf_result = torch_reshape(input_tensor_tf, target_shape)

    # Convert TensorFlow result to numpy for comparison
    tf_result_np = tf_result.numpy()

    # Check if the results match
    are_equal = np.allclose(torch_result.numpy(), tf_result_np)

    # Print results
    print("Input tensor (PyTorch):")
    print(input_tensor_torch)
    print("Target shape:", target_shape)
    print("PyTorch reshape result:")
    print(torch_result)
    print("TensorFlow reshape result:")
    print(tf_result_np)
    print("Are the results equal?", are_equal)

# Run the test
test_reshape()
