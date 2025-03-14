import unittest
import torch
import tensorflow as tf
import numpy as np
from util.torch_to_tf import torch_cat


class TestCat(unittest.TestCase):
    def test_cat_dim0(self):
        """Test concatenation along dimension 0"""
        a = tf.constant([[1, 2, 3], [4, 5, 6]])
        b = tf.constant([[7, 8, 9], [10, 11, 12]])

        torch_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        torch_b = torch.tensor([[7, 8, 9], [10, 11, 12]])

        s1 = torch_cat([a, b], dim=0)
        torch_s1 = torch.cat([torch_a, torch_b], dim=0)

        # print("s1 (cat axis=0):\n", s1.numpy())
        # print("torch_s1 (cat axis=0):\n", torch_s1.numpy())
        
        self.assertTrue(np.allclose(s1.numpy(), torch_s1.numpy()))

    def test_cat_dim1(self):
        """Test concatenation along dimension 1"""
        a = tf.constant([[1, 2, 3], [4, 5, 6]])
        b = tf.constant([[7, 8, 9], [10, 11, 12]])

        torch_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        torch_b = torch.tensor([[7, 8, 9], [10, 11, 12]])

        s2 = torch_cat([a, b], dim=1)
        torch_s2 = torch.cat([torch_a, torch_b], dim=1)

        # print("\n")
        # print("s2 (cat axis=1):\n", s2.numpy())
        # print("torch_s2 (cat axis=1):\n", torch_s2.numpy())
        
        self.assertTrue(np.allclose(s2.numpy(), torch_s2.numpy()))

    def test_cat_default_dim(self):
        """Test concatenation with default dimension"""
        a = tf.constant([[1, 2, 3], [4, 5, 6]])
        b = tf.constant([[7, 8, 9], [10, 11, 12]])

        torch_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        torch_b = torch.tensor([[7, 8, 9], [10, 11, 12]])

        torch_s3 = torch.cat([torch_a, torch_b])
        s3 = torch_cat([a, b])

        # print("s3 (cat axis=None):\n", torch_s3.numpy())
        # print("s3 (cat axis=None):\n", s3.numpy())
        # print("s3.shape (cat axis=None):\n", s3.numpy().shape)
        
        self.assertTrue(np.allclose(s3.numpy(), torch_s3.numpy()))


if __name__ == '__main__':
    unittest.main()
