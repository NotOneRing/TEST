import unittest
import torch
import tensorflow as tf
import numpy as np

class TestPermShuffle(unittest.TestCase):
    def setUp(self):
        # Set the size for permutation
        self.sz = 10
    
    def test_torch_randperm(self):
        """Test torch.randperm function"""
        # Generate random permutation using torch
        perm = torch.randperm(self.sz)
        # print("torch.randperm = ", perm)
        
        # Verify the shape
        self.assertEqual(perm.shape[0], self.sz)
        
        # Verify that all numbers from 0 to sz-1 are present
        sorted_perm = torch.sort(perm)[0]
        expected = torch.arange(self.sz)

        # print("sorted_perm = ", sorted_perm)
        # print("expected = ", expected)

        # print("sorted_perm.dtype = ", sorted_perm.dtype)
        # print("expected.dtype = ", expected.dtype)

        self.assertTrue(torch.all(sorted_perm == expected))
    
    def test_tf_random_shuffle(self):
        """Test tf.random.shuffle function"""
        # Generate random permutation using tensorflow
        # perm = tf.random.shuffle(tf.range(self.sz))
        from util.torch_to_tf import torch_randperm
        perm = torch_randperm(self.sz)
        # print("tf.random.shuffle = ", perm)
        
        # Verify the shape
        self.assertEqual(perm.shape[0], self.sz)
        
        # Verify that all numbers from 0 to sz-1 are present
        sorted_perm = tf.sort(perm)
        expected = tf.range(self.sz)

        # print("sorted_perm = ", sorted_perm)
        # print("expected = ", expected)

        # print("sorted_perm.dtype = ", sorted_perm.dtype)
        # print("expected.dtype = ", expected.dtype)
        expected = tf.cast(expected, tf.int64)
        self.assertTrue(tf.reduce_all(tf.equal(sorted_perm, expected)))

if __name__ == '__main__':
    unittest.main()
