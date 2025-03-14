import torch
import torch.nn as nn
import numpy as np
import unittest
import tensorflow as tf
from util.torch_to_tf import nn_Softplus


class TestSoftplus(unittest.TestCase):
    def test_Softplus(self):
        """
        Test that the PyTorch Softplus and TensorFlow Softplus implementations
        produce the same results for the same input.
        """
        # create a layer of Softplus
        softplus = nn.Softplus()

        # input tensor
        input_tensor = torch.tensor([-1.0, 0.0, 1.0])

        # apply softplus activation
        output = softplus(input_tensor)
        # print(output)

        # create a layer of Softplus
        softplus = nn_Softplus()

        # input tensor
        input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

        # apply softplus activation
        output_tf = softplus(input_tensor_tf)
        # print(output_tf)

        # Assert that the PyTorch and TensorFlow implementations produce the same results
        self.assertTrue(np.allclose(output, output_tf))


if __name__ == "__main__":
    unittest.main()
