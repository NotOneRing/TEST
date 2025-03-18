import tensorflow as tf
import torch
import unittest
from util.torch_to_tf import torch_dot


class TestTorchDot(unittest.TestCase):
    def test_torch_dot(self):
        # Create TensorFlow tensors
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])

        # Calculate dot product using the torch_dot function
        dot_product = torch_dot(a, b)
        # print(dot_product)

        # Create equivalent PyTorch tensors
        a_torch = torch.tensor(a.numpy())
        b_torch = torch.tensor(b.numpy())

        # Calculate dot product using PyTorch's native dot function
        torch_dot_product = torch.dot(a_torch, b_torch)
        # print(torch_dot_product)

        # Assert that both dot products are equal
        self.assertEqual(
            dot_product.numpy(), 
            torch_dot_product.numpy(), 
            "The tensorflow output is not equivalent to the torch one"
        )


if __name__ == "__main__":
    unittest.main()
