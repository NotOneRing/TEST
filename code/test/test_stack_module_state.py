import unittest
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from torch.func import stack_module_state
from util.torch_to_tf import nn_Linear, torch_func_stack_module_state, torch_tensor_transpose


class TestStackModuleState(unittest.TestCase):
    """
    Test case for comparing PyTorch's stack_module_state with TensorFlow implementation.
    """
    
    def setUp(self):
        """
        Set up test environment before each test method.
        Define model classes and initialize test variables.
        """
        # Define the simple PyTorch model
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 1)
        
        # Define the simple TensorFlow model
        class tf_SimpleNet(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.fc = nn_Linear(2, 1)
                
            def call(self, x):
                return self.fc(x)
        
        self.SimpleNet = SimpleNet
        self.tf_SimpleNet = tf_SimpleNet
        
        # Initialize result containers
        self.result1 = []
        self.result2 = []
        self.tf_result1 = []
        self.tf_result2 = []

    def test_single_model_stack(self):
        """
        Test stacking parameters and buffers of a single model instance.
        Compares PyTorch and TensorFlow implementations.
        """
        # Create single PyTorch model
        temp = [self.SimpleNet()]
        
        # Stack parameters and buffers for PyTorch model
        stacked_params1, stacked_buffers1 = stack_module_state(temp)
        
        # Collect stacked parameters
        self.result1 = []
        for k, v in stacked_params1.items():
            self.result1.append(v)
            
        # # Print debug information
        # print("Stacked Parameters Shape:", {k: v.shape for k, v in stacked_params1.items()})
        # print("Stacked Buffers:", stacked_buffers1)
        
        # Create single TensorFlow model
        temp_tf = [self.tf_SimpleNet()]
        
        # Initialize TensorFlow model with PyTorch weights
        for i, network in enumerate(temp_tf):
            _ = network(tf.constant(np.random.randn(1, 2).astype(np.float32)))
            
            if isinstance(temp[i].fc, nn.Linear):
                network.fc.trainable_weights[0].assign(temp[i].fc.weight.detach().numpy().T)  # kernel
                network.fc.trainable_weights[1].assign(temp[i].fc.bias.detach().numpy())     # bias
        
        # Stack parameters and buffers for TensorFlow model
        tf_stacked_params1, tf_stacked_buffers1 = torch_func_stack_module_state(temp_tf)
        
        # print("Stacked Parameters Shape:", {k: v.shape for k, v in tf_stacked_params1.items()})
        # print("Stacked Buffers:", tf_stacked_buffers1)
        
        # Collect stacked parameters with appropriate transposition for kernels
        self.tf_result1 = []
        for k, v in tf_stacked_params1.items():
            if 'kernel' in k:
                self.tf_result1.append(torch_tensor_transpose(v, 1, 2))
            else:
                self.tf_result1.append(v)
        
        # Assert that PyTorch and TensorFlow results match
        for i in range(len(self.result1)):
            self.assertTrue(
                np.allclose(self.result1[i].detach().numpy(), self.tf_result1[i].numpy()),
                f"Single model parameter at index {i} does not match between PyTorch and TensorFlow"
            )
            # print(f"Parameter {i} match: {np.allclose(self.result1[i].detach().numpy(), self.tf_result1[i].numpy())}")

    def test_multiple_models_stack(self):
        """
        Test stacking parameters and buffers of multiple model instances.
        Compares PyTorch and TensorFlow implementations.
        """
        # Create multiple PyTorch model instances
        models = [self.SimpleNet() for _ in range(3)]
        
        # Stack the parameters and buffers of these models
        stacked_params2, stacked_buffers2 = stack_module_state(models)
        
        # print("Stacked Parameters Shape:", {k: v.shape for k, v in stacked_params2.items()})
        # print("Stacked Buffers:", stacked_buffers2)
        
        # Collect stacked parameters
        self.result2 = []
        for k, v in stacked_params2.items():
            self.result2.append(v)
        
        # Create multiple TensorFlow model instances
        models_tf = [self.tf_SimpleNet() for _ in range(3)]
        
        # Initialize TensorFlow models with PyTorch weights
        for i, network in enumerate(models_tf):
            _ = network(tf.constant(np.random.randn(1, 2).astype(np.float32)))
            
            if isinstance(models[i].fc, nn.Linear):
                network.fc.trainable_weights[0].assign(models[i].fc.weight.detach().numpy().T)  # kernel
                network.fc.trainable_weights[1].assign(models[i].fc.bias.detach().numpy())     # bias
        
        # Stack the parameters and buffers of these TensorFlow models
        tf_stacked_params2, tf_stacked_buffers2 = torch_func_stack_module_state(models_tf)
        
        # print("Stacked Parameters Shape:", {k: v.shape for k, v in tf_stacked_params2.items()})
        # print("Stacked Buffers:", tf_stacked_buffers2)
        
        # Collect stacked parameters with appropriate transposition for kernels
        self.tf_result2 = []
        for k, v in tf_stacked_params2.items():
            # print("type(v) = ", type(v))
            if 'kernel' in k:
                self.tf_result2.append(torch_tensor_transpose(v, 1, 2))
            else:
                self.tf_result2.append(v)
        
        # Assert that PyTorch and TensorFlow results match
        for i in range(len(self.result2)):
            self.assertTrue(
                np.allclose(self.result2[i].detach().numpy(), self.tf_result2[i].numpy()),
                f"Multiple model parameter at index {i} does not match between PyTorch and TensorFlow"
            )
            # print(f"Parameter {i} match: {np.allclose(self.result2[i].detach().numpy(), self.tf_result2[i].numpy())}")


if __name__ == "__main__":
    unittest.main()
