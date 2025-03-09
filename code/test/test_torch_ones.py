import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_ones


# Test function: compare the ones tensor in torch and tensorflow
def test_zeros_equivalence(shape):
    # TensorFlow: ones tensor
    tf_tensor = torch_ones(shape, dtype=tf.float32)

    # PyTorch: zeros tensor
    torch_tensor = torch.ones(shape, dtype=torch.float32)

    # compare TensorFlow and PyTorch tensors
    print(f"TensorFlow ones tensor:\n{tf_tensor.numpy()}")
    print(f"PyTorch ones tensor:\n{torch_tensor.numpy()}")

    # Check the output of tensorflow and torch are equivalent
    match = np.allclose(tf_tensor.numpy(), torch_tensor.numpy())
    print(f"Outputs match: {match}")




    # TensorFlow ones tensor
    tf_tensor = torch_ones(*shape, dtype=tf.float32)

    # PyTorch ones tensor
    torch_tensor = torch.ones(*shape, dtype=torch.float32)

    # compare tensor in TensorFlow and PyTorch
    print(f"TensorFlow ones tensor:\n{tf_tensor.numpy()}")
    print(f"PyTorch ones tensor:\n{torch_tensor.numpy()}")

    # check if their output are equivalent
    match = np.allclose(tf_tensor.numpy(), torch_tensor.numpy())
    print(f"Outputs match: {match}")



    return match

# run testing
shape = (2, 3)  # set testing shape

test_zeros_equivalence(shape)





