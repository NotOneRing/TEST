import unittest
import torch
import torch.nn as nn
from torch.func import functional_call

import numpy as np

from copy import deepcopy

from util.torch_to_tf import torch_func_functional_call


class MyModel(nn.Module):
    """
    A simple linear model for testing functional_call.
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.linear(x)


import tensorflow as tf
from util.torch_to_tf import nn_Linear
class MyTFModel(tf.keras.Model):
    """
    A simple linear model for testing functional_call.
    """
    def __init__(self):
        super().__init__()
        self.linear = nn_Linear(3, 1)
    
    def call(self, x):
        return self.linear(x)





class TestTorchFuncFunctionalCall(unittest.TestCase):
    """
    Test case for torch.func.functional_call functionality.
    This tests the ability to call a model with explicit parameters and buffers.
    """
    def setUp(self):
        """
        Set up test environment before each test method.
        Initialize model and input data.
        """
        # Define a simple model
        self.model = MyModel()
        
        # Get model's parameters and buffers
        self.params_and_buffers = {
            **dict(self.model.named_parameters()), 
            **dict(self.model.named_buffers())
        }
        
        # Create input tensor
        self.x = torch.randn(2, 3)

        self.tf_x = tf.convert_to_tensor(self.x.numpy())

        self.tf_model = MyTFModel()
        _ = self.tf_model(self.tf_x)

        self.tf_model.linear.model.trainable_weights[0].assign(self.model.linear.weight.detach().numpy().T)  # kernel
        self.tf_model.linear.model.trainable_weights[1].assign(self.model.linear.bias.detach().numpy())     # bias


    def test_functional_call_basic(self):
        """
        Test basic functionality of torch.func.functional_call.
        Verifies that functional_call produces the same output as direct model call.
        """
        # Use functional_call to call model
        functional_output = functional_call(self.model, self.params_and_buffers, (self.x,))
        
        # Use direct call for comparison
        direct_output = self.model(self.x)
        
        # Verify outputs are identical
        self.assertTrue(torch.allclose(functional_output, direct_output))
        
    def test_functional_call_modified_params(self):
        """
        Test functional_call with modified parameters.
        Verifies that changing parameters affects the output without changing the original model.
        """
        # Create a copy of parameters and modify them
        modified_params = {k: v.clone() for k, v in self.params_and_buffers.items()}
        
        # Modify the weights - double the linear layer weights
        if 'linear.weight' in modified_params:
            modified_params['linear.weight'] = modified_params['linear.weight'] * 2.0
        
        # Call with modified parameters
        modified_output = functional_call(self.model, modified_params, (self.x,))
        
        # Call with original parameters
        original_output = functional_call(self.model, self.params_and_buffers, (self.x,))
        
        # Verify outputs are different
        self.assertFalse(torch.allclose(modified_output, original_output))
        
        # Verify original model is unchanged
        direct_output = self.model(self.x)
        self.assertTrue(torch.allclose(direct_output, original_output))


        tf_modified_params = deepcopy(self.tf_model.trainable_variables)
        
        # Call with original parameters
        tf_original_output = torch_func_functional_call(self.tf_model, self.tf_model.trainable_variables, (self.tf_x,))

        # print("1: tf_modified_params = ", tf_modified_params)
        # Call with modified parameters
        for target_var in tf_modified_params:
            if "kernel" in target_var.name:
                target_var.assign(target_var * 2)
        # print("2: tf_modified_params = ", tf_modified_params)

        tf_modified_output = torch_func_functional_call(self.tf_model, tf_modified_params, (self.tf_x,))

        tf_direct_output = self.tf_model( self.tf_x )

        # print("modified_output.detach().numpy() = ", modified_output.detach().numpy())
        # print("tf_modified_output.numpy() = ", tf_modified_output.numpy())

        self.assertTrue( np.allclose(modified_output.detach().numpy(), tf_modified_output.numpy(), atol=1e-3) )
        self.assertTrue( np.allclose(original_output.detach().numpy(), tf_original_output.numpy(), atol=1e-3) )

        # print("tf_direct_output.numpy() = ", tf_direct_output.numpy())
        # print("tf_original_output.numpy() = ", tf_original_output.numpy())
        self.assertTrue( np.allclose(tf_direct_output.numpy(), tf_original_output.numpy(), atol=1e-3) )


if __name__ == '__main__':
    unittest.main()
