import torch
import torch.nn as nn
import numpy as np
import unittest
import tensorflow as tf
from util.torch_to_tf import nn_Identity


class TestIdentity(unittest.TestCase):
    def test_Identity(self):
        # createa a layer of Identity
        identity = nn.Identity()

        # input tensor
        input_tensor = torch.tensor([-1.0, 0.0, 1.0])

        # apply identity activation
        output = identity(input_tensor)
        # print(output)

        # creatae a layer of Identity
        identity = nn_Identity()

        # input tensor
        input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

        # apply Identity
        output_tf = identity(input_tensor_tf)
        # print(output_tf)

        # Check if PyTorch and TensorFlow implementations produce the same result
        self.assertTrue(np.allclose(output, output_tf))


if __name__ == '__main__':
    unittest.main()
