import unittest
import torch
import tensorflow as tf
from util.torch_to_tf import torch_tensor_expand
import numpy as np

class TestExpand(unittest.TestCase):
    def test_expand(self):
        # create a 1D PyTorch Tensor
        tensor = torch.tensor([1, 2, 3])

        # expand to a 3x3 matrix using tuple arguments
        expanded_tensor_1 = tensor.expand(3, 3)
        
        # expand to a 3x3 matrix using list argument
        expanded_tensor_2 = tensor.expand([3, 3])

        # create a 1D TensorFlow Tensor
        tensor_tf = tf.constant([1, 2, 3])

        # expand to a 3x3 matrix using list argument
        expanded_tensor_tf1 = torch_tensor_expand(tensor_tf, [3, 3])
        
        # expand to a 3x3 matrix using multiple arguments
        expanded_tensor_tf2 = torch_tensor_expand(tensor_tf, 3, 3)

        # Assert that PyTorch and TensorFlow implementations produce the same results
        self.assertTrue(np.allclose(expanded_tensor_1.numpy(), expanded_tensor_tf1.numpy()))
        self.assertTrue(np.allclose(expanded_tensor_2.numpy(), expanded_tensor_tf2.numpy()))
        
        # Additional assertions to verify the expected shape and values
        self.assertEqual(expanded_tensor_1.shape, (3, 3))
        self.assertEqual(expanded_tensor_2.shape, (3, 3))
        self.assertEqual(expanded_tensor_tf1.shape, (3, 3))
        self.assertEqual(expanded_tensor_tf2.shape, (3, 3))

if __name__ == '__main__':
    unittest.main()

