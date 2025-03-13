import unittest
from util.torch_to_tf import torch_tensor_long
from util.func_pytest import np_to_tf, np_to_torch, gen_3d_int
import tensorflow as tf
import torch
import numpy as np

class TestLong(unittest.TestCase):
    def test_long(self):
        test_case = gen_3d_int()[0].astype(np.float32)
        tf_tensor = np_to_tf(test_case)
        torch_tensor = np_to_torch(test_case)

        # print("tf_tensor.dtype = ", tf_tensor.dtype)
        # print("torch_tensor.dtype = ", torch_tensor.dtype)

        tf_tensor = torch_tensor_long(tf_tensor)
        torch_tensor = torch_tensor.long()
        self.assertEqual(torch_tensor.dtype, torch.long)
        self.assertEqual(tf_tensor.dtype, tf.int64)


if __name__ == '__main__':
    unittest.main()
