import unittest
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from util.torch_to_tf import nn_GELU


class TestGELU(unittest.TestCase):
    def test_GELU(self):
        """
        Test that PyTorch and TensorFlow GELU implementations produce similar results.
        """
        # create a layer of GELU (PyTorch)
        gelu = nn.GELU()

        # input tensor
        input_tensor = torch.tensor([-1.0, 0.0, 1.0])

        # apply gelu activation
        output = gelu(input_tensor)
        # print(output)

        # create a layer of GELU (TensorFlow)
        gelu = nn_GELU()

        # input tensor
        input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

        # apply gelu activation
        output_tf = gelu(input_tensor_tf)
        # print(output_tf)

        # Assert that PyTorch and TensorFlow implementations produce similar results
        self.assertTrue(np.allclose(output, output_tf))


if __name__ == "__main__":
    unittest.main()
