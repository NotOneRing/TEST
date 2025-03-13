import unittest
import tensorflow as tf
import numpy as np
import torch
from util.torch_to_tf import torch_mean


class TestMean(unittest.TestCase):
    def setUp(self):
        self.np_array = np.array([[1, 2], [3, 4]]).astype(np.float32)
        self.tf_test_tensor = tf.convert_to_tensor( self.np_array )
        self.torch_test_tensor = torch.tensor( self.np_array )
        
    def test_tensorflow_torch_with_dim(self):
        tf_result = torch_mean(self.tf_test_tensor, dim=1)

        expected = tf.convert_to_tensor([1.5, 3.5])

        tf.debugging.assert_near(tf_result, expected, rtol=1e-5)
        
        torch_result = torch.mean(self.torch_test_tensor, dim=1)

        self.assertTrue( np.allclose(tf_result.numpy(), torch_result.numpy()) )


if __name__ == '__main__':
    unittest.main()


