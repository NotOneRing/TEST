import unittest
from util.torch_to_tf import torch_tensor_float
from util.func_pytest import np_to_tf, np_to_torch, gen_3d_int
import tensorflow as tf
import torch


class TestFloat(unittest.TestCase):
    def test_float(self):
        test_case = gen_3d_int()[0]
        tf_tensor = np_to_tf(test_case)
        torch_tensor = np_to_torch(test_case)

        # print("torch_tensor.dtype = ", torch_tensor.dtype)
        # print("tf_tensor.dtype = ", tf_tensor.dtype)

        tf_tensor = torch_tensor_float(tf_tensor)
        torch_tensor = torch_tensor.float()

        self.assertEqual(torch_tensor.dtype, torch.float32)
        self.assertEqual(tf_tensor.dtype, tf.float32)


if __name__ == '__main__':
    unittest.main()




