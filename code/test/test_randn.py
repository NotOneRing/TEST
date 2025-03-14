import torch
import unittest
import tensorflow as tf
from util.torch_to_tf import torch_randn


class TestRandn(unittest.TestCase):
    def test_randn(self):
        # create 2x3 standard normal distribution random tensor
        tensor = torch.randn(2, 3)
        self.assertEqual(tensor.shape, (2, 3))
        self.assertTrue(torch.isfinite(tensor).all())
        # print("tensor = ", tensor)

        tensor = torch.randn([2, 3])
        self.assertEqual(tensor.shape, (2, 3))
        self.assertTrue(torch.isfinite(tensor).all())
        # print("tensor = ", tensor)
        
        # create 2x3 standard normal distribution random tensor with tensorflow
        tensor_tf = torch_randn(2, 3)
        self.assertEqual(tensor_tf.shape, (2, 3))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(tensor_tf)))
        # print("tensor_tf = ", tensor_tf)
        
        tensor_tf = torch_randn([2, 3])
        self.assertEqual(tensor_tf.shape, (2, 3))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(tensor_tf)))
        # print("tensor_tf = ", tensor_tf)


if __name__ == '__main__':
    unittest.main()
