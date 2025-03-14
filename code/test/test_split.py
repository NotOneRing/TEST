import unittest
import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_split


class TestSplit(unittest.TestCase):
    """
    Test class for torch_split function from util.torch_to_tf module.
    Tests the compatibility between PyTorch's split and the TensorFlow implementation.
    """

    def test_split_equal_parts(self):
        """
        Test case 1: Split a tensor into equal parts along dimension 0.
        Verifies that torch_split produces the same result as torch.split when
        splitting a tensor into equal-sized chunks.
        """
        # PyTorch tensor
        pytorch_tensor = torch.randn(6, 4)  # Shape: (6, 4)
        pytorch_split = torch.split(pytorch_tensor, 4, dim=0)  # Split into parts along dim 0

        # TensorFlow tensor
        tf_tensor = tf.convert_to_tensor(pytorch_tensor.numpy())
        tf_split = torch_split(tf_tensor, 4, dim=0)  # Split into parts along axis 0

        # Verify the results have the same structure
        self.assertEqual(len(pytorch_split), len(tf_split))
        
        # Compare each part
        for i, (pytorch_part, tf_part) in enumerate(zip(pytorch_split, tf_split)):
            self.assertTrue(
                np.allclose(pytorch_part.detach().numpy(), tf_part.numpy(), atol=1e-5),
                f"Part {i+1} values do not match"
            )

    def test_split_different_sizes(self):
        """
        Test case 2: Split a tensor with different sizes for each part.
        Verifies that torch_split produces the same result as torch.split when
        splitting a tensor into chunks of different sizes.
        """
        # PyTorch tensor
        pytorch_tensor = torch.randn(10, 4)  # Shape: (10, 4)
        pytorch_split = torch.split(pytorch_tensor, [3, 3, 4], dim=0)  # Split into parts of size 3, 3, and 4 along dim 0

        # TensorFlow tensor
        tf_tensor = tf.convert_to_tensor(pytorch_tensor.numpy())
        tf_split = torch_split(tf_tensor, [3, 3, 4], dim=0)  # Split into parts of size 3, 3, and 4 along axis 0

        # Verify the results have the same structure
        self.assertEqual(len(pytorch_split), len(tf_split))
        
        # Compare each part
        for i, (pytorch_part, tf_part) in enumerate(zip(pytorch_split, tf_split)):
            self.assertTrue(
                np.allclose(pytorch_part.detach().numpy(), tf_part.numpy(), atol=1e-5),
                f"Part {i+1} values do not match"
            )

    def test_split_along_dimension_1(self):
        """
        Test case 3: Split a tensor along dimension 1.
        Verifies that torch_split produces the same result as torch.split when
        splitting a tensor along a non-zero dimension.
        """
        # PyTorch tensor
        pytorch_tensor = torch.randn(6, 4)  # Shape: (6, 4)
        pytorch_split = torch.split(pytorch_tensor, 2, dim=1)  # Split into 2 parts along dim 1 (4 elements, 2 per part)

        # TensorFlow tensor
        tf_tensor = tf.convert_to_tensor(pytorch_tensor.numpy())
        tf_split = torch_split(tf_tensor, 2, dim=1)  # Split into 2 parts along axis 1

        # Verify the results have the same structure
        self.assertEqual(len(pytorch_split), len(tf_split))
        
        # Compare each part
        for i, (pytorch_part, tf_part) in enumerate(zip(pytorch_split, tf_split)):
            self.assertTrue(
                np.allclose(pytorch_part.detach().numpy(), tf_part.numpy(), atol=1e-5),
                f"Part {i+1} values do not match"
            )


if __name__ == "__main__":
    unittest.main()
