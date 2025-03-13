import torch
import tensorflow as tf
import numpy as np
import unittest

from util.torch_to_tf import torch_flip


class TestTorchFlip(unittest.TestCase):
    
    def test_case_1(self):
        """Test flip operation on tensor with shape (2, 3, 4) along dimensions [0, 2]."""
        input_array = np.random.randint(0, 10, (2, 3, 4))
        tf_input = tf.constant(input_array)
        torch_input = torch.tensor(input_array)

        dims = [0, 2]
        tf_output = torch_flip(tf_input, dims)
        torch_output = torch.flip(torch_input, dims)

        # print("tf_output.shape = ", tf_output.shape)

        self.assertTrue(np.array_equal(tf_output.numpy(), torch_output.numpy()))

    def test_case_2(self):
        """Test flip operation on tensor with shape (3, 3) along dimension [0]."""
        input_array = np.random.randint(0, 10, (3, 3))
        tf_input = tf.constant(input_array)
        torch_input = torch.tensor(input_array)

        dims = [0]  # Flip along the first dimension
        tf_output = torch_flip(tf_input, dims)
        torch_output = torch.flip(torch_input, dims)

        self.assertTrue(np.array_equal(tf_output.numpy(), torch_output.numpy()))

    def test_case_3(self):
        """Test flip operation on tensor with shape (1, 5, 2) along dimension [1]."""
        input_array = np.random.randint(0, 10, (1, 5, 2))
        tf_input = tf.constant(input_array)
        torch_input = torch.tensor(input_array)

        dims = [1]  # Flip along the second dimension
        tf_output = torch_flip(tf_input, dims)
        torch_output = torch.flip(torch_input, dims)

        self.assertTrue(np.array_equal(tf_output.numpy(), torch_output.numpy()))

    def test_case_4(self):
        """Test flip operation on tensor with shape (4, 1, 3) along dimension [2]."""
        input_array = np.random.randint(0, 10, (4, 1, 3))
        tf_input = tf.constant(input_array)
        torch_input = torch.tensor(input_array)

        dims = [2]  # Flip along the third dimension
        tf_output = torch_flip(tf_input, dims)
        torch_output = torch.flip(torch_input, dims)

        self.assertTrue(np.array_equal(tf_output.numpy(), torch_output.numpy()))

    def test_case_5(self):
        """Test flip operation on tensor with shape (3, 1, 5, 2) along dimensions [0, 3]."""
        input_array = np.random.randint(0, 10, (3, 1, 5, 2))
        tf_input = tf.constant(input_array)
        torch_input = torch.tensor(input_array)

        dims = [0, 3]  # Flip along the first and last dimension
        tf_output = torch_flip(tf_input, dims)
        torch_output = torch.flip(torch_input, dims)

        self.assertTrue(np.array_equal(tf_output.numpy(), torch_output.numpy()))


if __name__ == "__main__":
    unittest.main()
