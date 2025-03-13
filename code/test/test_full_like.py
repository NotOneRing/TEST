import unittest
from util.torch_to_tf import torch_tensor, torch_full_like

import numpy as np
import tensorflow as tf
import torch


class TestFullLike(unittest.TestCase):
    def test_full_like(self):
        np_arr = np.array(range(9)).reshape(3, 3).astype(np.float32)

        tensor_torch = torch.tensor(np_arr)
        tensor_tf = torch_tensor(np_arr)

        tf_tensor_full_like = torch_full_like(tensor_tf, 3)
        torch_tensor_full_like = torch.full_like(tensor_torch, 3)

        # print("tf_tensor_full_like = ", tf_tensor_full_like)
        # print("torch_tensor_full_like = ", torch_tensor_full_like)

        self.assertTrue(np.allclose(tf_tensor_full_like.numpy(), torch_tensor_full_like.numpy()))


if __name__ == '__main__':
    unittest.main()
