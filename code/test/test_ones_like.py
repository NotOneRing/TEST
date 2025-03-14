import unittest
from util.torch_to_tf import torch_tensor, torch_ones_like

import numpy as np
import tensorflow as tf
import torch


class TestOnesLike(unittest.TestCase):
    def test_ones_like(self):
        np_arr = np.array(range(9)).reshape(3, 3).astype(np.float32)

        # .astype(np.float32)

        tensor_torch = torch.tensor(np_arr)
        tensor_tf = torch_tensor(np_arr)

        tf_tensor_ones_like = torch_ones_like(tensor_tf)
        # print("tf_tensor_ones_like = ", tf_tensor_ones_like)

        torch_tensor_one_like = torch.ones_like(tensor_torch)
        # print("torch_tensor_one_like = ", torch_tensor_one_like)

        assert np.allclose(tf_tensor_ones_like.numpy(), torch_tensor_one_like.numpy())


if __name__ == "__main__":
    unittest.main()
