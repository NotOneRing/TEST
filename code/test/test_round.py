import torch
import tensorflow as tf
import unittest
import numpy as np

from util.torch_to_tf import torch_round


class TestRound(unittest.TestCase):
    def test_round(self):
        # Create a tensor of float type
        tensor = torch.tensor([1.1, 2.5, 3.7, -1.4])

        # round off
        rounded_tensor = torch.round(tensor)

        # print(rounded_tensor)

        # Create a tensor of float type
        tensor = tf.constant([1.1, 2.5, 3.7, -1.4])

        # round off
        tf_rounded_tensor = torch_round(tensor)

        # print(tf_rounded_tensor)

        # Assert that PyTorch and TensorFlow implementations give the same results
        self.assertTrue(np.allclose(rounded_tensor, tf_rounded_tensor))


if __name__ == '__main__':
    unittest.main()
