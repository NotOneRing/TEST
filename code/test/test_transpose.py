import unittest
import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_tensor_transpose


class TestTranspose(unittest.TestCase):
    """
    Test class for tensor transpose operations between PyTorch and TensorFlow.
    Tests the equivalence of transpose operations in both frameworks.
    """
    def test_transpose_operations(self):
        """
        Test transpose operations with different tensor shapes and dimension pairs.
        Verifies that torch.transpose and the TensorFlow equivalent function
        produce the same results.
        """
        # Test cases with different shapes and dimension pairs
        test_cases = [
            {"shape": (2, 3), "dim0": 0, "dim1": 1},
            {"shape": (2, 3, 4), "dim0": 0, "dim1": 2},
            {"shape": (5, 4, 3, 2), "dim0": 1, "dim1": 3},
        ]

        for i, case in enumerate(test_cases):
            shape = case["shape"]
            dim0, dim1 = case["dim0"], case["dim1"]

            # Create random tensor
            torch_tensor = torch.rand(shape)
            tf_tensor = tf.convert_to_tensor(torch_tensor.numpy())

            # PyTorch transpose
            torch_transposed = torch.transpose(torch_tensor, dim0, dim1)

            # TensorFlow transpose
            tf_transposed = torch_tensor_transpose(tf_tensor, dim0, dim1)

            # Convert back to NumPy to compare
            torch_transposed_np = torch_transposed.numpy()
            tf_transposed_np = tf_transposed.numpy()

            # Test case information for debugging
            test_info = f"Test case {i + 1}: Original shape: {shape}, Transposed shape: {torch_transposed_np.shape}"
            
            # Assert that the results are equal within tolerance
            self.assertTrue(
                np.allclose(torch_transposed_np, tf_transposed_np),
                f"{test_info} - PyTorch and TensorFlow results do not match"
            )
            # print(f"Test case {i + 1}: Passed!")
            # print(f"Original shape: {shape}, Transposed shape: {torch_transposed_np.shape}")


if __name__ == "__main__":
    unittest.main()
