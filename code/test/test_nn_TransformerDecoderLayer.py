import unittest
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.torch_to_tf import nn_TransformerDecoderLayer, torch_tensor


class TestTransformerDecoderLayer(unittest.TestCase):
    """
    Test case for comparing PyTorch and TensorFlow implementations of TransformerDecoderLayer.
    This test ensures that the TensorFlow implementation produces the same outputs as PyTorch
    when initialized with the same weights.
    """

    def setUp(self):
        """
        Set up the test environment by initializing random seeds, model parameters,
        and creating test input data.
        """
        # Set random seeds to ensure reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        tf.random.set_seed(42)

        # Model parameters
        self.d_model = 4
        self.nhead = 2
        self.dim_feedforward = 16
        self.dropout = 0

        # Create PyTorch TransformerDecoderLayer
        self.decoder_layer_torch = nn.TransformerDecoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout
        )
        self.decoder_layer_torch.eval()

        # Create TensorFlow TransformerDecoderLayer
        self.decoder_layer_tf = nn_TransformerDecoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout
        )

        # Input data with batch=1
        self.torch_tgt = torch.tensor([
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.5, 0.6, 0.7, 0.8]]
        ])  # (tgt_len=2, batch_size=1, d_model=4)

        self.torch_memory = torch.tensor([
            [[0.9, 1.0, 1.1, 1.2]],
            [[1.3, 1.4, 1.5, 1.6]]
        ])  # (memory_len=2, batch_size=1, d_model=4)

        # Convert PyTorch tensors to TensorFlow tensors
        self.tf_tgt = torch_tensor(np.array([
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.5, 0.6, 0.7, 0.8]]
        ]))  # (tgt_len=2, batch_size=1, d_model=4)

        self.tf_memory = torch_tensor(np.array([
            [[0.9, 1.0, 1.1, 1.2]],
            [[1.3, 1.4, 1.5, 1.6]]
        ]))  # (memory_len=2, batch_size=1, d_model=4)

        # Initialize TensorFlow layer with PyTorch weights
        self._initialize_weights()

    def _initialize_attention(self, attention_torch, attention_tf, d_model):
        """
        Initialize TensorFlow's MultiheadAttention with weights from PyTorch's MultiheadAttention.
        
        Args:
            attention_torch: PyTorch MultiheadAttention module
            attention_tf: TensorFlow MultiheadAttention module
            d_model: Dimension of the model
        """
        query_weight_torch = attention_torch.in_proj_weight
        query_bias_torch = attention_torch.in_proj_bias
        output_weight_torch = attention_torch.out_proj.weight
        output_bias_torch = attention_torch.out_proj.bias

        # # Use random input to trigger weight initialization
        # dummy_query = tf.random.normal((1, 1, d_model))
        # dummy_key = tf.random.normal((1, 1, d_model))
        # dummy_value = tf.random.normal((1, 1, d_model))
        # attention_tf(dummy_query, dummy_key, dummy_value)
        
        # Set parameters from PyTorch to TensorFlow
        attention_tf.query_dense.kernel.assign(
            tf.convert_to_tensor(query_weight_torch[:d_model].detach().numpy().T)
        )
        attention_tf.query_dense.bias.assign(
            tf.convert_to_tensor(query_bias_torch[:d_model].detach().numpy())
        )
        
        attention_tf.key_dense.kernel.assign(
            tf.convert_to_tensor(query_weight_torch[d_model:2*d_model].detach().numpy().T)
        )
        attention_tf.key_dense.bias.assign(
            tf.convert_to_tensor(query_bias_torch[d_model:2*d_model].detach().numpy())
        )
        
        attention_tf.value_dense.kernel.assign(
            tf.convert_to_tensor(query_weight_torch[2*d_model:].detach().numpy().T)
        )
        attention_tf.value_dense.bias.assign(
            tf.convert_to_tensor(query_bias_torch[2*d_model:].detach().numpy())
        )
        
        attention_tf.output_dense.kernel.assign(
            tf.convert_to_tensor(output_weight_torch.detach().numpy().T)
        )
        attention_tf.output_dense.bias.assign(
            tf.convert_to_tensor(output_bias_torch.detach().numpy())
        )

    def _initialize_weights(self):
        """
        Initialize all weights of the TensorFlow model with the PyTorch model weights.
        This includes attention layers, linear layers, and normalization layers.
        """
        # Initialize layers
        _ = self.decoder_layer_tf(self.tf_tgt, self.tf_memory)

        self._initialize_attention(self.decoder_layer_torch.self_attn, 
                                  self.decoder_layer_tf.self_attn, 
                                  self.d_model)
        
        self._initialize_attention(self.decoder_layer_torch.multihead_attn, 
                                  self.decoder_layer_tf.multihead_attn, 
                                  self.d_model)

        # Initialize linear layers
        self.decoder_layer_tf.linear1.trainable_weights[0].assign(
            self.decoder_layer_torch.linear1.weight.detach().numpy().T
        )
        self.decoder_layer_tf.linear1.trainable_weights[1].assign(
            self.decoder_layer_torch.linear1.bias.detach().numpy()
        )
        self.decoder_layer_tf.linear2.trainable_weights[0].assign(
            self.decoder_layer_torch.linear2.weight.detach().numpy().T
        )
        self.decoder_layer_tf.linear2.trainable_weights[1].assign(
            self.decoder_layer_torch.linear2.bias.detach().numpy()
        )

        # Initialize normalization layers
        self.decoder_layer_tf.norm1.trainable_weights[0].assign(
            self.decoder_layer_torch.norm1.weight.detach().numpy()
        )
        self.decoder_layer_tf.norm1.trainable_weights[1].assign(
            self.decoder_layer_torch.norm1.bias.detach().numpy()
        )
        self.decoder_layer_tf.norm2.trainable_weights[0].assign(
            self.decoder_layer_torch.norm2.weight.detach().numpy()
        )
        self.decoder_layer_tf.norm2.trainable_weights[1].assign(
            self.decoder_layer_torch.norm2.bias.detach().numpy()
        )
        self.decoder_layer_tf.norm3.trainable_weights[0].assign(
            self.decoder_layer_torch.norm3.weight.detach().numpy()
        )
        self.decoder_layer_tf.norm3.trainable_weights[1].assign(
            self.decoder_layer_torch.norm3.bias.detach().numpy()
        )

    def test_full_forward_pass(self):
        """
        Test the full forward pass of the TransformerDecoderLayer.
        Compares the output of PyTorch and TensorFlow implementations.
        """
        # Forward pass in PyTorch
        torch_output = self.decoder_layer_torch(self.torch_tgt, self.torch_memory)
        
        # Forward pass in TensorFlow
        tf_output = self.decoder_layer_tf(self.tf_tgt, self.tf_memory, training=False)
        
        # print("torch_output = ", torch_output)
        # print("tf_output = ", tf_output)

        # Compare outputs
        self.assertTrue(np.allclose(
            tf_output.numpy(), 
            torch_output.detach().numpy(), 
            atol=1e-3
        ), "Full forward pass outputs do not match")

    def test_self_attention(self):
        """
        Test the self-attention component of the TransformerDecoderLayer.
        Compares the output of PyTorch and TensorFlow implementations.
        """
        tgt_mask = None
        tgt_key_padding_mask = None
        
        # Test based on norm_first configuration
        if self.decoder_layer_tf.norm_first:
            # PyTorch implementation
            torch_tgt2 = self.decoder_layer_torch.self_attn(
                self.decoder_layer_torch.norm1(self.torch_tgt), 
                self.decoder_layer_torch.norm1(self.torch_tgt), 
                self.decoder_layer_torch.norm1(self.torch_tgt),
                attn_mask=tgt_mask, 
                key_padding_mask=tgt_key_padding_mask
            )[0]
            
            # TensorFlow implementation
            tf_tgt2 = self.decoder_layer_tf.self_attn(
                self.decoder_layer_tf.norm1(self.tf_tgt), 
                self.decoder_layer_tf.norm1(self.tf_tgt), 
                self.decoder_layer_tf.norm1(self.tf_tgt),
                attn_mask=tgt_mask, 
                key_padding_mask=tgt_key_padding_mask
            )[0]
            
            # Compare self-attention outputs
            self.assertTrue(np.allclose(
                tf_tgt2.numpy(), 
                torch_tgt2.detach().numpy(), 
                atol=1e-3
            ), "Self-attention outputs do not match (norm_first=True)")
            
            # Apply dropout and residual connection
            torch_tgt = self.torch_tgt + self.decoder_layer_torch.dropout1(torch_tgt2)
            tf_tgt = self.tf_tgt + self.decoder_layer_tf.dropout1(tf_tgt2)
            
            # Compare results after dropout and residual connection
            self.assertTrue(np.allclose(
                tf_tgt.numpy(), 
                torch_tgt.detach().numpy(), 
                atol=1e-3
            ), "Self-attention with dropout outputs do not match (norm_first=True)")
        else:
            # PyTorch implementation
            torch_tgt2 = self.decoder_layer_torch.self_attn(
                self.torch_tgt, 
                self.torch_tgt, 
                self.torch_tgt, 
                attn_mask=tgt_mask, 
                key_padding_mask=tgt_key_padding_mask
            )[0]
            
            # TensorFlow implementation
            tf_tgt2 = self.decoder_layer_tf.self_attn(
                self.tf_tgt, 
                self.tf_tgt, 
                self.tf_tgt, 
                attn_mask=tgt_mask, 
                key_padding_mask=tgt_key_padding_mask
            )[0]
            
            # Compare self-attention outputs
            self.assertTrue(np.allclose(
                tf_tgt2.numpy(), 
                torch_tgt2.detach().numpy(), 
                atol=1e-2
            ), "Self-attention outputs do not match (norm_first=False)")
            
            # Ensure shapes match
            self.assertEqual(
                tf_tgt2.numpy().shape, 
                torch_tgt2.detach().numpy().shape, 
                "Shape mismatch in self-attention outputs"
            )
            
            # Apply dropout, residual connection, and normalization
            tf_tgt = tf.cast(self.tf_tgt, tf.float32)
            tf_tgt2 = tf.cast(tf_tgt2, tf.float32)
            
            tf_tgt = self.decoder_layer_tf.norm1(
                tf_tgt + self.decoder_layer_tf.dropout1(tf_tgt2)
            )
            torch_tgt = self.decoder_layer_torch.norm1(
                self.torch_tgt + self.decoder_layer_torch.dropout1(torch_tgt2)
            )
            
            # Compare results after normalization
            self.assertTrue(np.allclose(
                tf_tgt.numpy(), 
                torch_tgt.detach().numpy(), 
                atol=1e-2
            ), "Self-attention with normalization outputs do not match (norm_first=False)")

    def test_cross_attention(self):
        """
        Test the cross-attention component of the TransformerDecoderLayer.
        Compares the output of PyTorch and TensorFlow implementations.
        """
        memory_mask = None
        memory_key_padding_mask = None
        
        # First run self-attention to get the updated target
        tgt_mask = None
        tgt_key_padding_mask = None
        
        if self.decoder_layer_tf.norm_first:
            # Self-attention
            torch_tgt2 = self.decoder_layer_torch.self_attn(
                self.decoder_layer_torch.norm1(self.torch_tgt), 
                self.decoder_layer_torch.norm1(self.torch_tgt), 
                self.decoder_layer_torch.norm1(self.torch_tgt),
                attn_mask=tgt_mask, 
                key_padding_mask=tgt_key_padding_mask
            )[0]
            torch_tgt = self.torch_tgt + self.decoder_layer_torch.dropout1(torch_tgt2)
            
            # Cross-attention
            torch_tgt2 = self.decoder_layer_torch.multihead_attn(
                self.decoder_layer_torch.norm2(torch_tgt), 
                self.decoder_layer_torch.norm2(self.torch_memory), 
                self.decoder_layer_torch.norm2(self.torch_memory),
                attn_mask=memory_mask, 
                key_padding_mask=memory_key_padding_mask
            )[0]
            
            # Convert PyTorch tensors to TensorFlow for comparison
            tf_tgt2 = tf.convert_to_tensor(torch_tgt2.detach().numpy().astype(np.float32))
            tf_tgt = tf.convert_to_tensor(torch_tgt.detach().numpy().astype(np.float32))
            
            # Apply dropout and residual connection
            tf_tgt = tf_tgt + self.decoder_layer_tf.dropout2(tf_tgt2)
            torch_tgt = torch_tgt + self.decoder_layer_torch.dropout2(torch_tgt2)
            
            # Compare results
            self.assertTrue(np.allclose(
                tf_tgt.numpy(), 
                torch_tgt.detach().numpy(), 
                atol=1e-3
            ), "Cross-attention outputs do not match (norm_first=True)")
        else:
            # Self-attention
            torch_tgt2 = self.decoder_layer_torch.self_attn(
                self.torch_tgt, 
                self.torch_tgt, 
                self.torch_tgt, 
                attn_mask=tgt_mask, 
                key_padding_mask=tgt_key_padding_mask
            )[0]
            torch_tgt = self.decoder_layer_torch.norm1(
                self.torch_tgt + self.decoder_layer_torch.dropout1(torch_tgt2)
            )
            
            # Cross-attention
            torch_tgt2 = self.decoder_layer_torch.multihead_attn(
                torch_tgt, 
                self.torch_memory, 
                self.torch_memory, 
                attn_mask=memory_mask, 
                key_padding_mask=memory_key_padding_mask
            )[0]
            
            # Convert PyTorch tensors to TensorFlow for comparison
            tf_tgt2 = tf.convert_to_tensor(torch_tgt2.detach().numpy().astype(np.float32))
            tf_tgt = tf.convert_to_tensor(torch_tgt.detach().numpy().astype(np.float32))
            
            # Apply normalization, dropout, and residual connection
            tf_tgt = self.decoder_layer_tf.norm2(tf_tgt + self.decoder_layer_tf.dropout2(tf_tgt2))
            torch_tgt = self.decoder_layer_torch.norm2(torch_tgt + self.decoder_layer_torch.dropout2(torch_tgt2))
            
            # Compare results
            self.assertTrue(np.allclose(
                tf_tgt.numpy(), 
                torch_tgt.detach().numpy(), 
                atol=1e-3
            ), "Cross-attention outputs do not match (norm_first=False)")

    def test_feedforward_network(self):
        """
        Test the feedforward network component of the TransformerDecoderLayer.
        Compares the output of PyTorch and TensorFlow implementations.
        """
        # # First run self-attention and cross-attention to get the updated target
        tgt_mask = None
        tgt_key_padding_mask = None
        memory_mask = None
        memory_key_padding_mask = None

        # Convert PyTorch tensors to TensorFlow tensors
        tf_tgt = self.tf_tgt
        tf_memory = self.tf_memory
        
        torch_tgt = self.torch_tgt
        torch_memory = self.torch_memory

        # 1. Self-Attention (Decoder)
        if self.decoder_layer_tf.norm_first:
            tf_tgt2 = self.decoder_layer_tf.self_attn(self.decoder_layer_tf.norm1(tf_tgt), self.decoder_layer_tf.norm1(tf_tgt), self.decoder_layer_tf.norm1(tf_tgt),
                                    attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
            torch_tgt2 = self.decoder_layer_torch.self_attn(self.decoder_layer_torch.norm1(torch_tgt), self.decoder_layer_torch.norm1(torch_tgt), self.decoder_layer_torch.norm1(torch_tgt),
                                    attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]



            tf_tgt = tf_tgt + self.decoder_layer_tf.dropout1(tf_tgt2)
            torch_tgt = torch_tgt + self.decoder_layer_tf.dropout1(torch_tgt2)

            assert np.allclose( tf_tgt2.numpy(), torch_tgt2.detach().numpy() , atol=1e-3 )
            assert tf_tgt2.numpy().shape == torch_tgt2.detach().numpy().shape, "shape does not match"
            assert np.allclose( tf_tgt.numpy(), torch_tgt.detach().numpy() , atol=1e-3 )
            assert tf_tgt.numpy().shape == torch_tgt.detach().numpy().shape, "shape does not match"

        else:
            tf_tgt2 = self.decoder_layer_tf.self_attn(tf_tgt, tf_tgt, tf_tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
            torch_tgt2 = self.decoder_layer_torch.self_attn(torch_tgt, torch_tgt, torch_tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]

            tf_tgt = tf.cast( tf_tgt, tf.float32 )
            tf_tgt2 = tf.cast( tf_tgt2, tf.float32 )

            tf_tgt = self.decoder_layer_tf.norm1(tf_tgt + self.decoder_layer_tf.dropout1(tf_tgt2))
            torch_tgt = self.decoder_layer_torch.norm1(torch_tgt + self.decoder_layer_torch.dropout1(torch_tgt2))

            assert np.allclose( tf_tgt2.numpy(), torch_tgt2.detach().numpy() , atol=1e-2 )
            assert tf_tgt2.numpy().shape == torch_tgt2.detach().numpy().shape, "shape does not match"
            assert np.allclose( tf_tgt.numpy(), torch_tgt.detach().numpy() , atol=1e-2 )
            assert tf_tgt.numpy().shape == torch_tgt.detach().numpy().shape, "shape does not match"


        # 2. Cross-Attention (Encoder-Decoder)
        if self.decoder_layer_tf.norm_first:
            tf_tgt2 = self.decoder_layer_tf.multihead_attn(self.decoder_layer_tf.norm2(tf_tgt), self.decoder_layer_tf.norm2(tf_memory), self.decoder_layer_tf.norm2(tf_memory),
                                        attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]

            torch_tgt2 = self.decoder_layer_torch.multihead_attn(self.decoder_layer_torch.norm2(torch_tgt), self.decoder_layer_torch.norm2(torch_memory), self.decoder_layer_torch.norm2(torch_memory),
                                        attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]

            tf_tgt2 = tf.convert_to_tensor( torch_tgt2.detach().numpy().astype(np.float32) )
            tf_tgt = tf.convert_to_tensor( torch_tgt.detach().numpy().astype(np.float32) )

            tf_tgt = tf_tgt + self.decoder_layer_tf.dropout2(tf_tgt2)
            torch_tgt = torch_tgt + self.decoder_layer_torch.dropout2(torch_tgt2)

            assert np.allclose( tf_tgt2.numpy(), torch_tgt2.detach().numpy() , atol=1e-2 )
            assert tf_tgt2.numpy().shape == torch_tgt2.detach().numpy().shape, "shape does not match"
            assert np.allclose( tf_tgt.numpy(), torch_tgt.detach().numpy() , atol=1e-2 )
            assert tf_tgt.numpy().shape == torch_tgt.detach().numpy().shape, "shape does not match"

        else:
            tf_tgt2 = self.decoder_layer_tf.multihead_attn(tf_tgt, tf_memory, tf_memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
            torch_tgt2 = self.decoder_layer_torch.multihead_attn(torch_tgt, torch_memory, torch_memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]

            tf_tgt2 = tf.convert_to_tensor( torch_tgt2.detach().numpy().astype(np.float32) )
            tf_tgt = tf.convert_to_tensor( torch_tgt.detach().numpy().astype(np.float32) )

        
            tf_tgt = self.decoder_layer_tf.norm2(tf_tgt + self.decoder_layer_tf.dropout2(tf_tgt2))
            torch_tgt = self.decoder_layer_torch.norm2(torch_tgt + self.decoder_layer_torch.dropout2(torch_tgt2))

            assert np.allclose( tf_tgt2.numpy(), torch_tgt2.detach().numpy() , atol=1e-2 )
            assert tf_tgt2.numpy().shape == torch_tgt2.detach().numpy().shape, "shape does not match"
            assert np.allclose( tf_tgt.numpy(), torch_tgt.detach().numpy() , atol=1e-2 )
            assert tf_tgt.numpy().shape == torch_tgt.detach().numpy().shape, "shape does not match"


        # 3. Feedforward Network (FFN)
        if self.decoder_layer_tf.norm_first:
            # if training:
            tf_tgt2 = self.decoder_layer_tf.linear2(self.decoder_layer_tf.dropout(self.decoder_layer_tf.activation(self.decoder_layer_tf.linear1(self.decoder_layer_tf.norm3(tf_tgt)))))
            tf_tgt = tf_tgt + self.decoder_layer_tf.dropout3(tf_tgt2)

            torch_tgt2 = self.decoder_layer_torch.linear2( self.decoder_layer_torch.dropout( self.decoder_layer_torch.activation( self.decoder_layer_torch.linear1( self.decoder_layer_torch.norm3( torch_tgt ) ) ) ) )
            torch_tgt = torch_tgt + self.decoder_layer_torch.dropout3(torch_tgt2)

            assert np.allclose( tf_tgt2.numpy(), torch_tgt2.detach().numpy() , atol=1e-2 )
            assert tf_tgt2.numpy().shape == torch_tgt2.detach().numpy().shape, "shape does not match"
            assert np.allclose( tf_tgt.numpy(), torch_tgt.detach().numpy() , atol=1e-2 )
            assert tf_tgt.numpy().shape == torch_tgt.detach().numpy().shape, "shape does not match"

        else:
            tf_tgt2 = self.decoder_layer_tf.linear2(self.decoder_layer_tf.dropout(self.decoder_layer_tf.activation(self.decoder_layer_tf.linear1(tf_tgt))))
            tf_tgt = self.decoder_layer_tf.norm3(tf_tgt + self.decoder_layer_tf.dropout3(tf_tgt2))

            torch_tgt2 = self.decoder_layer_torch.linear2( self.decoder_layer_torch.dropout( self.decoder_layer_torch.activation( self.decoder_layer_torch.linear1(torch_tgt) ) ) )
            torch_tgt = self.decoder_layer_torch.norm3( torch_tgt + self.decoder_layer_torch.dropout3( torch_tgt2 ) )

            assert np.allclose( tf_tgt2.numpy(), torch_tgt2.detach().numpy() , atol=1e-2 )
            assert tf_tgt2.numpy().shape == torch_tgt2.detach().numpy().shape, "shape does not match"
            assert np.allclose( tf_tgt.numpy(), torch_tgt.detach().numpy() , atol=1e-2 )
            assert tf_tgt.numpy().shape == torch_tgt.detach().numpy().shape, "shape does not match"



        # Forward pass in PyTorch
        torch_output = self.decoder_layer_torch(self.torch_tgt, self.torch_memory)
        

        assert np.allclose( tf_tgt.numpy(), torch_tgt.detach().numpy() , atol=1e-4 )
        assert np.allclose( tf_tgt2.numpy(), torch_tgt2.detach().numpy() , atol=1e-4 )
        assert np.allclose( tf_tgt.numpy(), torch_output.detach().numpy() , atol=1e-4 )


if __name__ == "__main__":
    unittest.main()
