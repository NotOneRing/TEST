import torch
import tensorflow as tf
import numpy as np

# # define torch_tensor_transpose function
# def torch_tensor_transpose(input, dim0, dim1):
#     dim_lens = len(input.shape)
#     perm = list(range(dim_lens))
#     temp = perm[dim0]
#     perm[dim0] = perm[dim1]
#     perm[dim1] = temp
#     return tf.transpose(input, perm=perm)

from util.torch_to_tf import torch_tensor_transpose


def test_transpose():

    # test cases
    test_cases = [
        {"shape": (2, 3), "dim0": 0, "dim1": 1},
        {"shape": (2, 3, 4), "dim0": 0, "dim1": 2},
        {"shape": (5, 4, 3, 2), "dim0": 1, "dim1": 3},
    ]

    for i, case in enumerate(test_cases):
        shape = case["shape"]
        dim0, dim1 = case["dim0"], case["dim1"]

        # create random tensor
        torch_tensor = torch.rand(shape)
        tf_tensor = tf.convert_to_tensor(torch_tensor.numpy())

        # PyTorch transpose
        torch_transposed = torch.transpose(torch_tensor, dim0, dim1)

        # TensorFlow transpose
        tf_transposed = torch_tensor_transpose(tf_tensor, dim0, dim1)

        # convert back to NumPy to compare
        torch_transposed_np = torch_transposed.numpy()
        tf_transposed_np = tf_transposed.numpy()

        # print testing results
        assert np.allclose(torch_transposed_np, tf_transposed_np), f"Test case {i + 1} failed!"
        print(f"Test case {i + 1}: Passed!")
        print(f"Original shape: {shape}, Transposed shape: {torch_transposed_np.shape}")


test_transpose()


