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
        attention_tf = nn_MultiheadAttention(num_heads=self.num_heads, d_model=self.d_model, batch_first=True)
        

        # Use random input to trigger weight initialization
        dummy_query = tf.random.normal((1, 1, self.d_model))
        dummy_key = tf.random.normal((1, 1, self.d_model))
        dummy_value = tf.random.normal((1, 1, self.d_model))
        attention_tf(dummy_query, dummy_key, dummy_value)
        


        # for name, param in attention_torch.named_parameters():
        #     print(f"1-Torch: Parameter name: {name}")
        #     print(f"1-Torch: Value: {param.data}\n")


        # for var in attention_tf.variables:
        #     print(f"1-Tensorflow: Parameter name: {var.name}")
        #     print(f"1-Tensorflow: Value: {var.numpy()}\n")



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


        # for name, param in attention_torch.named_parameters():
        #     print(f"2-Torch: Parameter name: {name}")
        #     print(f"2-Torch: Value: {param.data}\n")


        # for var in attention_tf.variables:
        #     print(f"2-Tensorflow: Parameter name: {var.name}")
        #     print(f"2-Tensorflow: Value: {var.numpy()}\n")


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

        self.assertTrue(
            output_torch_np.shape == output_tf_np.shape,
            f"Output tensors shape don't match. output_torch_np.shape {output_torch_np.shape} and output_tf_np.shape {output_tf_np.shape}"
        )


        # Assert attention weights are close
        self.assertTrue(
            np.allclose(attention_weights_torch_np, attention_weights_tf_np),
            f"Attention weight tensors don't match. Mean absolute difference: {attention_weights_diff}"
        )

        self.assertTrue(
            attention_weights_torch_np.shape == attention_weights_tf_np.shape,
            f"Output tensors shape don't match. attention_weights_torch_np.shape {attention_weights_torch_np.shape} and attention_weights_tf_np.shape {attention_weights_tf_np.shape}"
        )

    def test_multihead_attention_output2(self):
        """Test if PyTorch and TensorFlow MultiheadAttention produce similar outputs"""

        from util.torch_to_tf import torch_tensor

        self.batch_size = 1
        self.seq_len = 2
        self.d_model = 4
        self.num_heads = 2

        torch_tgt = torch.tensor([[[0.1, 0.2, 0.3, 0.4]],
                            [[0.5, 0.6, 0.7, 0.8]]])  # (tgt_len=2, batch_size=1, d_model=4)

        torch_memory = torch.tensor([[[0.9, 1.0, 1.1, 1.2]],
                            [[1.3, 1.4, 1.5, 1.6]]])  # (memory_len=2, batch_size=1, d_model=4)

        tf_tgt = torch_tensor( np.array([[[0.1, 0.2, 0.3, 0.4]],
                            [[0.5, 0.6, 0.7, 0.8]]]) )  # (tgt_len=2, batch_size=1, d_model=4)

        tf_memory = torch_tensor( np.array([[[0.9, 1.0, 1.1, 1.2]],
                            [[1.3, 1.4, 1.5, 1.6]]]) )  # (memory_len=2, batch_size=1, d_model=4)

        # print("torch_tgt.shape = ", torch_tgt.shape)
        # print("torch_memory.shape = ", torch_memory.shape)
        # print("tf_tgt.shape = ", tf_tgt.shape)
        # print("tf_memory.shape = ", tf_memory.shape)

        # self.query_torch = torch.randn(self.batch_size, self.seq_len, self.d_model)
        # self.key_torch = torch.randn(self.batch_size, self.seq_len, self.d_model)
        # self.value_torch = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.query_torch = torch_tgt
        self.key_torch = torch_tgt
        self.value_torch = torch_tgt
                
        self.query_tf = tf_tgt
        self.key_tf = tf_tgt
        self.value_tf = tf_tgt


        # Create PyTorch's MultiheadAttention
        attention_torch = torch.nn.MultiheadAttention(
            embed_dim=self.d_model, 
            num_heads=self.num_heads, 
            batch_first=False
        )
        
        # Get PyTorch parameters
        query_weight_torch = attention_torch.in_proj_weight
        query_bias_torch = attention_torch.in_proj_bias
        output_weight_torch = attention_torch.out_proj.weight
        output_bias_torch = attention_torch.out_proj.bias
        
        # Initialize TensorFlow's MultiheadAttention
        attention_tf = nn_MultiheadAttention(num_heads=self.num_heads, d_model=self.d_model, batch_first=False)
        

        # Use random input to trigger weight initialization
        dummy_query = tf.random.normal((1, 1, self.d_model))
        dummy_key = tf.random.normal((1, 1, self.d_model))
        dummy_value = tf.random.normal((1, 1, self.d_model))
        attention_tf(dummy_query, dummy_key, dummy_value)
        


        # for name, param in attention_torch.named_parameters():
        #     print(f"1-Torch: Parameter name: {name}")
        #     print(f"1-Torch: Value: {param.data}\n")


        # for var in attention_tf.variables:
        #     print(f"1-Tensorflow: Parameter name: {var.name}")
        #     print(f"1-Tensorflow: Value: {var.numpy()}\n")



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


        # for name, param in attention_torch.named_parameters():
        #     print(f"2-Torch: Parameter name: {name}")
        #     print(f"2-Torch: Value: {param.data}\n")


        # for var in attention_tf.variables:
        #     print(f"2-Tensorflow: Parameter name: {var.name}")
        #     print(f"2-Tensorflow: Value: {var.numpy()}\n")


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
        
        
        # print("np.abs(output_torch_np - output_tf_np) = ", np.abs(output_torch_np - output_tf_np))


        # Assert outputs are close
        self.assertTrue(
            np.allclose(output_torch_np, output_tf_np, atol=1e-3),
            f"Output tensors don't match. Mean absolute difference: {output_diff}"
        )

        self.assertTrue(
            output_torch_np.shape == output_tf_np.shape,
            f"Output tensors shape don't match. output_torch_np.shape {output_torch_np.shape} and output_tf_np.shape {output_tf_np.shape}"
        )

        # Assert attention weights are close
        self.assertTrue(
            np.allclose(attention_weights_torch_np, attention_weights_tf_np, atol=1e-4),
            f"Attention weight tensors don't match. Mean absolute difference: {attention_weights_diff}"
        )

        self.assertTrue(
            attention_weights_torch_np.shape == attention_weights_tf_np.shape,
            f"Output tensors shape don't match. attention_weights_torch_np.shape {attention_weights_torch_np.shape} and attention_weights_tf_np.shape {attention_weights_tf_np.shape}"
        )


if __name__ == "__main__":
    unittest.main()



