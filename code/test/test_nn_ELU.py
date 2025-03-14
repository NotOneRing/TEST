import unittest
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from util.torch_to_tf import nn_ELU

class TestELU(unittest.TestCase):
    """
    Test case for comparing PyTorch's ELU activation with TensorFlow implementation.
    """
    
    def test_elu_activation(self):
        """
        Test that PyTorch and TensorFlow ELU activations produce the same output
        for the same input.
        """
        # create a layer of ELU activation in PyTorch
        elu = nn.ELU()

        # input tensor for PyTorch
        input_tensor = torch.tensor([-1.0, 0.0, 1.0])

        # apply ELU activation in PyTorch
        output = elu(input_tensor)
        
        # create a layer of ELU activation in TensorFlow
        elu_tf = nn_ELU()

        input_tensor_tf = tf.convert_to_tensor(input_tensor.numpy())

        # apply ELU activation in TensorFlow
        output_tf = elu_tf(input_tensor_tf)
        
        # Assert that the outputs are the same
        self.assertTrue(np.allclose(output, output_tf), 
                        f"PyTorch output: {output}, TensorFlow output: {output_tf}")


if __name__ == '__main__':
    unittest.main()
