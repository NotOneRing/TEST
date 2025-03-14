import unittest
import tensorflow as tf
import torch
import numpy as np


class TestTensorShape(unittest.TestCase):
    """
    Test class for comparing tensor reshaping operations in PyTorch and TensorFlow.
    """
    
    def setUp(self):
        """
        Set up the test data - a 3D numpy array that will be used for both frameworks.
        """
        # Create a 3D numpy array for testing
        self.arr = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
        
    def test_torch_reshape(self):
        """
        Test PyTorch tensor reshaping from 3D to 1D (flattened).
        """
        # Convert numpy array to PyTorch tensor
        torch_tensor = torch.tensor(self.arr)
        
        # Reshape the tensor to a flat array
        torch_tensor = torch_tensor.reshape(-1)
        
        # Verify the shape is correct (should be a 1D tensor with 8 elements)
        self.assertEqual(torch_tensor.shape, (8,))
        
        # Verify the values are preserved after reshaping
        expected_values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])

        # print("torch_tensor.dtype = ", torch_tensor.dtype)
        # print("expected_values.dtype = ", expected_values.dtype)

        self.assertTrue(torch.all(torch.eq(torch_tensor, expected_values)))
        
    def test_tensorflow_reshape(self):
        """
        Test TensorFlow tensor reshaping from 3D to 1D (flattened).
        """
        # Convert numpy array to TensorFlow tensor
        tf_tensor = tf.convert_to_tensor(self.arr)
        
        # Reshape the tensor to a flat array
        tf_tensor = tf.reshape(tf_tensor, -1)
        
        # Verify the shape is correct (should be a 1D tensor with 8 elements)
        self.assertEqual(tf_tensor.shape, (8,))
        
        # Verify the values are preserved after reshaping
        expected_values = tf.convert_to_tensor([1, 2, 3, 4, 5, 6, 7, 8])
        expected_values = tf.cast(expected_values, tf.int64)

        # print("tf_tensor.dtype = ", tf_tensor.dtype)
        # print("expected_values.dtype = ", expected_values.dtype)

        self.assertTrue(tf.reduce_all(tf.equal(tf_tensor, expected_values)))
        
    def test_reshape_equivalence(self):
        """
        Test that PyTorch and TensorFlow reshape operations produce equivalent results.
        """
        # Convert numpy array to tensors and reshape
        torch_tensor = torch.tensor(self.arr).reshape(-1)
        tf_tensor = tf.reshape(tf.convert_to_tensor(self.arr), -1)
        
        # Convert back to numpy for comparison
        torch_result = torch_tensor.numpy()
        tf_result = tf_tensor.numpy()
        
        # Verify both frameworks produce the same result
        np.testing.assert_array_equal(torch_result, tf_result)


if __name__ == '__main__':
    unittest.main()
