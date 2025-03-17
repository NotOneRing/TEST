import unittest
import tensorflow as tf
import torch
import numpy as np
from util.torch_to_tf import nn_MultiheadAttention


class TestMultiheadAttention(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        tf.random.set_seed(42)
        
        self.batch_size = 2
        self.seq_len = 5
        self.d_model = 16
        self.num_heads = 4
        
        self.query_torch = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.key_torch = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.value_torch = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        self.query_tf = tf.convert_to_tensor(self.query_torch.numpy())
        self.key_tf = tf.convert_to_tensor(self.key_torch.numpy())
        self.value_tf = tf.convert_to_tensor(self.value_torch.numpy())

    def test_multihead_attention_output(self):
        """Test if PyTorch and TensorFlow MultiheadAttention produce similar outputs"""
        # Create PyTorch's MultiheadAttention
        attention_torch = torch.nn.MultiheadAttention(
            embed_dim=self.d_model, 
            num_heads=self.num_heads, 
            batch_first=True
        )
        
        # Get PyTorch parameters
        query_weight_torch = attention_torch.in_proj_weight
        query_bias_torch = attention_torch.in_proj_bias
        output_weight_torch = attention_torch.out_proj.weight
        output_bias_torch = attention_torch.out_proj.bias
        
        # Initialize TensorFlow's MultiheadAttention
        attention_tf = nn_MultiheadAttention(num_heads=self.num_heads, d_model=self.d_model)
        
        # Use random input to trigger weight initialization
        dummy_query = tf.random.normal((1, 1, self.d_model))
        dummy_key = tf.random.normal((1, 1, self.d_model))
        dummy_value = tf.random.normal((1, 1, self.d_model))
        attention_tf(dummy_query, dummy_key, dummy_value)
        
        # Set parameters from PyTorch to TensorFlow
        attention_tf.query_dense.kernel.assign(
            tf.convert_to_tensor(query_weight_torch[:self.d_model].detach().numpy().T)
        )
        attention_tf.query_dense.bias.assign(
            tf.convert_to_tensor(query_bias_torch[:self.d_model].detach().numpy())
        )
        
        attention_tf.key_dense.kernel.assign(
            tf.convert_to_tensor(query_weight_torch[self.d_model:2*self.d_model].detach().numpy().T)
        )
        attention_tf.key_dense.bias.assign(
            tf.convert_to_tensor(query_bias_torch[self.d_model:2*self.d_model].detach().numpy())
        )
        
        attention_tf.value_dense.kernel.assign(
            tf.convert_to_tensor(query_weight_torch[2*self.d_model:].detach().numpy().T)
        )
        attention_tf.value_dense.bias.assign(
            tf.convert_to_tensor(query_bias_torch[2*self.d_model:].detach().numpy())
        )
        
        attention_tf.output_dense.kernel.assign(
            tf.convert_to_tensor(output_weight_torch.detach().numpy().T)
        )
        attention_tf.output_dense.bias.assign(
            tf.convert_to_tensor(output_bias_torch.detach().numpy())
        )
        
        # Run PyTorch's MultiheadAttention
        output_torch, attention_weights_torch = attention_torch(
            self.query_torch, self.key_torch, self.value_torch
        )
        
        # Run TensorFlow's MultiheadAttention
        output_tf, attention_weights_tf = attention_tf(
            self.query_tf, self.key_tf, self.value_tf
        )
        
        # Convert to NumPy
        output_torch_np = output_torch.detach().numpy()
        attention_weights_torch_np = attention_weights_torch.detach().numpy()
        
        output_tf_np = output_tf.numpy()
        attention_weights_tf_np = attention_weights_tf.numpy()
        
        # Calculate differences for debugging
        output_diff = np.mean(np.abs(output_torch_np - output_tf_np))
        attention_weights_diff = np.mean(np.abs(attention_weights_torch_np - attention_weights_tf_np))
        
        # Assert outputs are close
        self.assertTrue(
            np.allclose(output_torch_np, output_tf_np, atol=1e-6),
            f"Output tensors don't match. Mean absolute difference: {output_diff}"
        )
        
        # Assert attention weights are close
        self.assertTrue(
            np.allclose(attention_weights_torch_np, attention_weights_tf_np),
            f"Attention weight tensors don't match. Mean absolute difference: {attention_weights_diff}"
        )


if __name__ == "__main__":
    unittest.main()



