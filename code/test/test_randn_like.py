import unittest
from util.torch_to_tf import torch_tensor, torch_randn_like

import numpy as np
import tensorflow as tf
import torch


class TestRandnLike(unittest.TestCase):
    def test_randn_like(self):
        # Create a numpy array
        np_arr = np.array(range(9)).reshape(3, 3).astype(np.float32)
        # .astype(np.float32)

        # Convert to PyTorch and TensorFlow tensors
        tensor_torch = torch.tensor(np_arr)
        tensor_tf = torch_tensor(np_arr)

        # Apply randn_like to both tensors
        tf_tensor_randn_like = torch_randn_like(tensor_tf)
        torch_tensor_randn_like = torch.randn_like(tensor_torch)

        # # Print results (keeping original print statements for reference)
        # print("tf_tensor_randn_like = ", tf_tensor_randn_like)
        # print("torch_tensor_randn_like = ", torch_tensor_randn_like)

        # Assert that the shapes are the same
        self.assertEqual(tf_tensor_randn_like.shape, tensor_tf.shape)
        self.assertEqual(torch_tensor_randn_like.shape, tensor_torch.shape)

        # Assert that the dtypes are the same
        self.assertEqual(tf_tensor_randn_like.dtype, tensor_tf.dtype)
        self.assertEqual(torch_tensor_randn_like.dtype, tensor_torch.dtype)


if __name__ == '__main__':
    unittest.main()
