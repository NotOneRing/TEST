import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import unittest
from util.torch_to_tf import nn_ReLU

class TestReLU(unittest.TestCase):
    def test_ReLU(self):
        # create a layer of ReLU
        relu = nn.ReLU()

        # input tensor
        input_tensor = torch.tensor([-1.0, 0.0, 1.0])

        # apply relu activation
        output = relu(input_tensor)
        # print(output)

        # create a layer of ReLU
        relu = nn_ReLU()

        # input tensor
        input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

        # apply relu activation
        output_tf = relu(input_tensor_tf)
        # print(output_tf)

        # verify that PyTorch and TensorFlow implementations produce the same result
        self.assertTrue(np.allclose(output, output_tf))

if __name__ == '__main__':
    unittest.main()
