import torch
import tensorflow as tf
import numpy as np


from util.torch_to_tf import torch_zeros


# test function, compare zero tensor in tensorflow and torch
def test_zeros_equivalence(shape):
    # TensorFlow: zeros tensor
    tf_tensor = torch_zeros(shape, dtype=tf.float32)

    # PyTorch: zeros tensor
    torch_tensor = torch.zeros(shape, dtype=torch.float32)

    # compare tensor in TensorFlow and PyTorch
    print(f"TensorFlow zeros tensor:\n{tf_tensor.numpy()}")
    print(f"PyTorch zeros tensor:\n{torch_tensor.numpy()}")

    # check if their outputs are equivalent
    match = np.allclose(tf_tensor.numpy(), torch_tensor.numpy())
    print(f"Outputs match: {match}")




    # TensorFlow: zeros tensor
    tf_tensor = torch_zeros(*shape, dtype=tf.float32)

    # PyTorch: zeros tensor
    torch_tensor = torch.zeros(*shape, dtype=torch.float32)

    # compare output tensors in TensorFlow and PyTorch
    print(f"TensorFlow zeros tensor:\n{tf_tensor.numpy()}")
    print(f"PyTorch zeros tensor:\n{torch_tensor.numpy()}")

    # check if their outputs are equivalent
    match = np.allclose(tf_tensor.numpy(), torch_tensor.numpy())
    print(f"Outputs match: {match}")



    return match

# run test
shape = (2, 3)  # set testing shape

test_zeros_equivalence(shape)


