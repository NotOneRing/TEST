import unittest
import tensorflow as tf
import numpy as np
from util.torch_to_tf import _sum_rightmost

class TestSumRightmost(unittest.TestCase):
    """Test cases for the _sum_rightmost function."""
    
    def test_sum_rightmost_3d_tensor(self):
        """Test _sum_rightmost with a 3D tensor, summing the last 2 dimensions."""
        # Create a 3D tensor with shape [3, 3, 3]
        x = tf.convert_to_tensor(np.array(range(27)))
        x = tf.reshape(x, [3, 3, 3])
        
        # Sum the last two dimensions
        result = _sum_rightmost(x, 2)
        
        # Expected shape should be [3]
        self.assertEqual(result.shape, (3,))
        
        # Calculate expected values manually for verification
        expected = tf.reduce_sum(x, axis=[-2, -1])

        # print("result = ", result)
        # print("expected = ", expected)

        self.assertTrue( np.allclose(result.numpy(), expected.numpy(), atol=1e-6) )
        
    def test_sum_rightmost_4d_tensor(self):
        """Test _sum_rightmost with a 4D tensor, summing the last 2 dimensions."""
        # Create a 4D tensor with shape [2, 3, 4, 5]
        x = tf.random.normal([2, 3, 4, 5])
        
        # Sum the last two dimensions
        result = _sum_rightmost(x, 2)
        
        # Expected shape should be [2, 3]
        self.assertEqual(result.shape, (2, 3))
        
        # Calculate expected values manually for verification
        expected = tf.reduce_sum(x, axis=[-2, -1])

        # print("result = ", result)
        # print("expected = ", expected)

        self.assertTrue( np.allclose(result.numpy(), expected.numpy(), atol=1e-6) )
    
    def test_sum_rightmost_with_different_dims(self):
        """Test _sum_rightmost with different numbers of dimensions to sum."""
        # Create a 4D tensor
        x = tf.random.normal([2, 3, 4, 5])
        
        # Sum only the last dimension
        result_1d = _sum_rightmost(x, 1)
        self.assertEqual(result_1d.shape, (2, 3, 4))
        expected_1d = tf.reduce_sum(x, axis=[-1])

        self.assertTrue( np.allclose(result_1d.numpy(), expected_1d.numpy(), atol=1e-6) )

        # Sum the last three dimensions
        result_3d = _sum_rightmost(x, 3)
        self.assertEqual(result_3d.shape, (2,))
        expected_3d = tf.reduce_sum(x, axis=[-3, -2, -1])

        # print("result_3d = ", result_3d)
        # print("expected_3d = ", expected_3d)

        self.assertTrue( np.allclose(result_3d.numpy(), expected_3d.numpy(), atol=1e-6) )

if __name__ == '__main__':
    unittest.main()
