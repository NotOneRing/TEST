import torch
import torch.nn as nn
import numpy as np
import unittest
import tensorflow as tf
from util.torch_to_tf import nn_Tanh


class TestTanh(unittest.TestCase):
    def test_Tanh(self):
        # create a layer of Tanh
        tanh = nn.Tanh()

        # input tensor
        input_tensor = torch.tensor([-1.0, 0.0, 1.0])

        # create tanh activation
        output = tanh(input_tensor)
        # print(output)

        # creata a layer of Tanh
        tanh = nn_Tanh()

        # input tensor
        input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

        # apply tanh activation
        output_tf = tanh(input_tensor_tf)
        # print(output_tf)

        # verify that PyTorch and TensorFlow implementations produce the same results
        self.assertTrue(np.allclose(output, output_tf))


if __name__ == '__main__':
    unittest.main()
