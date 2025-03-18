import unittest
import tensorflow as tf
import numpy as np
import torch
from util.torch_to_tf import torch_tensor

class TestExpandDims(unittest.TestCase):
    def setUp(self):
        # Create the test array that will be used in all tests
        self.a = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])
    
    def test_numpy_expand_dims(self):
        # Test numpy's expand_dims functionality using [:, None]
        
        # print("self.a.shape = ", self.a.shape)

        expanded = self.a[:, None]
        
        # print("expanded.shape = ", expanded.shape)

        self.assertEqual(expanded.shape, (3, 1, 3))
        
        np.testing.assert_array_equal(expanded[:, 0, :], self.a)
    
    def test_torch_expand_dims(self):
        # Convert numpy array to torch tensor and test expand_dims
        b = torch.tensor(self.a)
        expanded_b = b[:, None]
        
        self.assertEqual(expanded_b.shape, (3, 1, 3))
        
        torch.testing.assert_close(expanded_b[:, 0, :], b)
    
    def test_torch_to_tf_expand_dims(self):
        # Test the custom torch_tensor function with expand_dims
        c = torch_tensor(self.a)
        expanded_c = c[:, None]
        
        self.assertEqual(tuple(expanded_c.shape), (3, 1, 3))
        
        tf.debugging.assert_equal(expanded_c[:, 0, :], c)

if __name__ == '__main__':
    unittest.main()


