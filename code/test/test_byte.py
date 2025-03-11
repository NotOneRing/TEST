import unittest
import tensorflow as tf
import torch
from util.torch_to_tf import torch_tensor_byte
from util.func_pytest import np_to_tf, np_to_torch, gen_3d_int

class TestByte(unittest.TestCase):
    def test_byte(self):
        test_case = gen_3d_int()[0]
        tf_tensor = np_to_tf(test_case)
        torch_tensor = np_to_torch(test_case)
        tf_tensor = torch_tensor_byte(tf_tensor)
        torch_tensor = torch_tensor.byte()
        self.assertEqual(torch_tensor.dtype, torch.uint8)
        self.assertEqual(tf_tensor.dtype, tf.uint8)

if __name__ == '__main__':
    unittest.main()






