import unittest
import numpy as np
import torch
import tensorflow as tf

from util.torch_to_tf import torch_where


class TestWhere(unittest.TestCase):
    """Test cases for torch_where function that verifies compatibility between PyTorch and TensorFlow."""

    def test_where_with_condition(self):
        """Test where function with condition: keep positive values and replace negative values with 0."""
        # Create a numpy array to test
        np_array = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])

        # Transform to torch.tensor and tf.Tensor
        torch_tensor = torch.tensor(np_array)
        tf_tensor = tf.convert_to_tensor(np_array)

        # Use where function: keep positive values and replace negative values by 0
        torch_result = torch.where(torch_tensor > 0, torch_tensor, 0)
        tf_result = torch_where(tf_tensor > 0, tf_tensor, 0)

        # Verify results match
        self.assertTrue(np.allclose(torch_result.numpy(), tf_result.numpy()))

    def test_where_all_false(self):
        """Test where function with a tensor of all False values."""
        # Create a tensor with all False values
        tensor = torch.tensor(
            [False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False])

        # Get indices of True values (should be empty)
        torch_result = torch.where(tensor)

        # Create equivalent TensorFlow tensor
        tf_tensor = tf.convert_to_tensor(np.array(
            [False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False]))

        # Get indices of True values using torch_where
        tf_result = torch_where(tensor)

        # print("2: tf_result = ", tf_result)
        # print("2: len(tf_result) = ", len(tf_result))
        # print("2: tf_result[0] = ", tf_result[0])

        # Verify results match
        self.assertTrue(np.allclose(torch_result[0].numpy(), tf_result[0].numpy()))

    def test_where_some_true(self):
        """Test where function with a tensor containing some True values."""
        # Create a tensor with some True values
        tensor = torch.tensor(
            [True, False, True, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False])

        # Get indices of True values
        torch_result = torch.where(tensor)

        # Create equivalent TensorFlow tensor
        tf_tensor = tf.convert_to_tensor(np.array(
            [True, False, True, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False]))

        # Get indices of True values using torch_where
        tf_result = torch_where(tf_tensor)

        # print("3: tf_result = ", tf_result)
        # print("3: len(tf_result) = ", len(tf_result))
        # print("3: tf_result[0] = ", tf_result[0])

        # Verify results match
        self.assertTrue(np.allclose(torch_result[0].numpy(), tf_result[0].numpy()))
        
        # Verify shape is correct
        self.assertEqual(torch_result[0].shape, tf_result[0].shape)

    def test_where_multidimensional(self):
        """Test where function with a multidimensional tensor."""
        # Create a 3D tensor with some True values
        tensor = torch.tensor(
            [True, False, True, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False]).reshape(2, 2, 10)

        # Get indices of True values
        torch_result = torch.where(tensor)

        # Create equivalent TensorFlow tensor
        tf_tensor = tf.reshape(tf.convert_to_tensor(np.array(
            [True, False, True, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False, False, False,
             False, False, False, False])), (2, 2, 10))

        # Get indices of True values using torch_where
        tf_result = torch_where(tf_tensor)

        # print("4: tf_result = ", tf_result)
        # print("4: len(tf_result) = ", len(tf_result))
        # print("4: tf_result[0] = ", tf_result[0])

        # Verify results match
        self.assertTrue(np.allclose(torch_result[0].numpy(), tf_result[0].numpy()))

        
        # Verify shape is correct
        self.assertEqual(torch_result[0].shape, tf_result[0].shape)


if __name__ == '__main__':
    unittest.main()
