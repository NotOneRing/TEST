import torch
import tensorflow as tf
import numpy as np
import unittest

from util.torch_to_tf import torch_tensor_repeat


class TestTensorRepeat(unittest.TestCase):
    def test_tensor_repeat_with_unpacked_repeats(self):
        """Test tensor repeat with unpacked repeats tuple."""
        # Create a random tensor in PyTorch
        torch_tensor = torch.tensor([[1, 2], [3, 4]])

        # Specify repeat pattern
        repeats = (2, 3, 4)

        # Repeat the tensor in PyTorch
        pytorch_result = torch_tensor.repeat(*repeats).numpy()

        # print("PyTorch result:\n", pytorch_result)

        # Convert PyTorch tensor to TensorFlow tensor
        tf_tensor = tf.convert_to_tensor(torch_tensor.numpy())

        # Repeat the tensor in TensorFlow
        tensorflow_result = torch_tensor_repeat(tf_tensor, *repeats).numpy()
        
        # print("TensorFlow result:\n", tensorflow_result)

        # Check if the results match
        self.assertTrue(np.array_equal(pytorch_result, tensorflow_result),
                        "The results do not match!")

    def test_tensor_repeat_with_packed_repeats(self):
        """Test tensor repeat with packed repeats tuple."""
        # Create a random tensor in PyTorch
        torch_tensor = torch.tensor([[1, 2], [3, 4]])

        # Specify repeat pattern
        repeats = (2, 3, 4)

        # Repeat the tensor in PyTorch
        pytorch_result = torch_tensor.repeat(repeats).numpy()

        # print("PyTorch result:\n", pytorch_result)
        # print("pytorch_result.shape = ", pytorch_result.shape)

        # Convert PyTorch tensor to TensorFlow tensor
        tf_tensor = tf.convert_to_tensor(torch_tensor.numpy())

        # Repeat the tensor in TensorFlow
        tensorflow_result = torch_tensor_repeat(tf_tensor, repeats).numpy()

        # print("TensorFlow result:\n", tensorflow_result)

        # Check if the results match
        self.assertTrue(np.array_equal(pytorch_result, tensorflow_result),
                        "The results do not match!")


if __name__ == '__main__':
    unittest.main()
