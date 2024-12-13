import torch
import tensorflow as tf
import numpy as np


from util.torch_to_tf import torch_tensor_repeat


# Test function to compare outputs
def test_tensor_repeat():
    # Create a random tensor in PyTorch
    torch_tensor = torch.tensor([[1, 2], [3, 4]])

    # Specify repeat pattern
    repeats = (2, 3, 4)

    # Repeat the tensor in PyTorch
    pytorch_result = torch_tensor.repeat(*repeats).numpy()

    print("pytorch_result.shape = ", pytorch_result.shape)

    # Convert PyTorch tensor to TensorFlow tensor
    tf_tensor = tf.convert_to_tensor(torch_tensor.numpy())

    # Repeat the tensor in TensorFlow
    tensorflow_result = torch_tensor_repeat(tf_tensor, *repeats).numpy()

    # Compare the results
    print("PyTorch result:\n", pytorch_result)
    print("TensorFlow result:\n", tensorflow_result)

    # Check if the results match
    if np.array_equal(pytorch_result, tensorflow_result):
        print("The results match!")
    else:
        print("The results do not match!")

# Run the test
test_tensor_repeat()


















