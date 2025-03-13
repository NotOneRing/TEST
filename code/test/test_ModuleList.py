import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import unittest

from util.torch_to_tf import nn_ModuleList


class TestModuleList(unittest.TestCase):
    
    def align_initialization(self, torch_modulelist, tf_modulelist):
        """Align initialization between PyTorch and TensorFlow module lists."""
        for torch_layer, tf_layer in zip(torch_modulelist, tf_modulelist):
            if isinstance(torch_layer, nn.Linear) and isinstance(tf_layer, tf.keras.layers.Dense):
                # Trigger weight initialization
                dummy_input = tf.random.normal([1, torch_layer.in_features])
                tf_layer(dummy_input)

                # Extract PyTorch's initialization weights and biases
                torch_weight = torch_layer.weight.detach().numpy()
                torch_bias = torch_layer.bias.detach().numpy() if torch_layer.bias is not None else None
                
                # Assign to the TensorFlow layer
                tf_layer.kernel.assign(torch_weight.T)  # transpose the weights
                if torch_bias is not None:
                    tf_layer.bias.assign(torch_bias)
                else:
                    print("torch_bias is None")

    def torch_modulelist_layer_func(self, module_list, input_tensor):
        """Test PyTorch module list."""
        outputs = []
        for module in module_list:
            input_tensor = module(input_tensor)
            outputs.append(input_tensor)
        return outputs

    def tf_modulelist_layer_func(self, module_list, input_tensor):
        """Test TensorFlow module list."""
        outputs = module_list(input_tensor)
        return outputs

    def test_modulelist_equivalence(self):
        """Test that PyTorch and TensorFlow ModuleList implementations are equivalent."""
        # Create the same input tensor
        torch_input = torch.randn(1, 10)
        tf_input = tf.convert_to_tensor(torch_input.detach().numpy(), dtype=tf.float32)
        
        # Create PyTorch and TensorFlow module lists
        torch_modulelist = nn.ModuleList([
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        ])

        tf_modulelist = nn_ModuleList([
            tf.keras.layers.Dense(20, activation=None),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(5, activation=None)
        ])

        # Align initialization
        self.align_initialization(torch_modulelist, tf_modulelist)
        
        # Get PyTorch's and TensorFlow's outputs
        torch_outputs = self.torch_modulelist_layer_func(torch_modulelist, torch_input)
        tf_outputs = self.tf_modulelist_layer_func(tf_modulelist, tf_input)
        # torch_outputs = torch_modulelist(torch_input)
        # tf_outputs = tf_modulelist(tf_input)
        
        # Ensure the number of output layers are consistent
        self.assertEqual(
            len(torch_outputs), 
            len(tf_outputs), 
            f"Mismatch in number of layers: {len(torch_outputs)} (torch) vs {len(tf_outputs)} (tf)"
        )
        
        # Compare the output of each layer
        for i, (torch_out, tf_out) in enumerate(zip(torch_outputs, tf_outputs)):
            # Convert to NumPy arrays
            torch_out_np = torch_out.detach().numpy()
            tf_out_np = tf_out.numpy()
            
            # Check both shapes are consistent
            self.assertEqual(
                torch_out_np.shape, 
                tf_out_np.shape,
                f"Layer {i} shape mismatch: Torch {torch_out_np.shape}, TF {tf_out_np.shape}"
            )
            print(f"Layer {i} shape match: {torch_out_np.shape}")
            
            # Check both values are consistent
            match = np.allclose(torch_out_np, tf_out_np, atol=1e-5)
            print(f"Layer {i} output match: {match}")
            
            self.assertTrue(match, f"Layer {i} outputs do not match")
            
            # Print output values of each layer (for debugging)
            print(f"Layer {i} Torch output:\n{torch_out_np}")
            print(f"Layer {i} TF output:\n{tf_out_np}")
            print("-" * 50)


if __name__ == '__main__':
    unittest.main()
