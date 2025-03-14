import unittest
import torch
import tensorflow as tf
import numpy as np

from util.func_pytest import gen_2d_int, np_to_torch, np_to_tf
from util.torch_to_tf import torch_triu


class TestTriu(unittest.TestCase):
    """
    Test class for comparing torch.triu and its TensorFlow implementation.
    """

    def setUp(self):
        """
        Set up test data before each test method.
        """
        # Generate a 2D integer matrix for testing
        self.matrix = gen_2d_int()[0]
        # Convert to torch tensor
        self.torch_matrix = np_to_torch(self.matrix)
        # Convert to tensorflow tensor
        self.tf_matrix = np_to_tf(self.matrix)

    def test_triu_diagonal_zero(self):
        """
        Test triu with diagonal=0 (main diagonal and upper triangular part).
        """
        # Extract the main diagonal and the upper triangular part of the matrix.
        triu_matrix = torch.triu(self.torch_matrix, diagonal=0)
        # output:
        # tensor([[1, 2, 3],
        #         [0, 5, 6],
        #         [0, 0, 9]])

        # Create a matrix with elements in the upper triangular part
        upper_triangle_matrix = torch_triu(self.tf_matrix, diagonal=0)

        # Compare results
        self.assertTrue(np.allclose(triu_matrix.numpy(), upper_triangle_matrix.numpy()))

    def test_triu_diagonal_positive(self):
        """
        Test triu with diagonal=1 (above main diagonal).
        """
        # diagonal is positive, extract from the upper triangular part above the main diagonal of the matrix.
        triu_matrix = torch.triu(self.torch_matrix, diagonal=1)
        # output:
        # tensor([[0, 2, 3],
        #         [0, 0, 6],
        #         [0, 0, 0]])

        upper_triangle_matrix = torch_triu(self.tf_matrix, diagonal=1)

        # Compare results
        self.assertTrue(np.allclose(triu_matrix.numpy(), upper_triangle_matrix.numpy()))

    def test_triu_diagonal_negative(self):
        """
        Test triu with diagonal=-1 (main diagonal and one below).
        """
        # diagonal is negative, extract from the lower triangular part below the main diagonal of the matrix.
        triu_matrix = torch.triu(self.torch_matrix, diagonal=-1)
        # output:
        # tensor([[1, 2, 3],
        #         [4, 5, 6],
        #         [0, 8, 9]])

        upper_triangle_matrix = torch_triu(self.tf_matrix, diagonal=-1)

        # Compare results
        self.assertTrue(np.allclose(triu_matrix.numpy(), upper_triangle_matrix.numpy()))


if __name__ == '__main__':
    unittest.main()
