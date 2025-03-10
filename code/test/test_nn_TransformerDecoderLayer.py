import numpy as np
# import tensorflow as tf
# import torch


# from util.torch_to_tf import nn_TransformerDecoderLayer


# # class nn_TransformerDecoderLayer(tf.keras.layers.Layer):
# #     def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, name="nn_TransformerDecoderLayer", **kwargs):

# #         # if OUTPUT_FUNCTION_HEADER:
# #         #     print("called nn_TransformerDecoderLayer __init__()")

# #         super(nn_TransformerDecoderLayer, self).__init__(name=name, **kwargs)
# #         self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
# #         self.cross_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
# #         self.ffn = tf.keras.Sequential([
# #             tf.keras.layers.Dense(dim_feedforward, activation=activation),
# #             tf.keras.layers.Dropout(dropout),
# #             tf.keras.layers.Dense(d_model),
# #         ])
# #         self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
# #         self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
# #         self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
# #         self.dropout1 = tf.keras.layers.Dropout(dropout)
# #         self.dropout2 = tf.keras.layers.Dropout(dropout)
# #         self.dropout3 = tf.keras.layers.Dropout(dropout)

# #     def call(self, tgt, memory, tgt_mask=None, memory_mask=None, training=None):
# #         # Self-attention on target
# #         tgt2 = self.self_attn(tgt, tgt, attention_mask=tgt_mask, training=training)
# #         tgt = tgt + self.dropout1(tgt2, training=training)
# #         tgt = self.norm1(tgt)

# #         # Cross-attention between target and memory
# #         tgt2 = self.cross_attn(tgt, memory, attention_mask=memory_mask, training=training)
# #         tgt = tgt + self.dropout2(tgt2, training=training)
# #         tgt = self.norm2(tgt)

# #         # Feedforward network
# #         tgt2 = self.ffn(tgt, training=training)
# #         tgt = tgt + self.dropout3(tgt2, training=training)
# #         tgt = self.norm3(tgt)

# #         return tgt





# # Parameters
# d_model = 8
# nhead = 2
# dim_feedforward = 16
# dropout = 0.1
# activation = "relu"
# n_layers = 2
# sequence_length = 5
# batch_size = 3

# # Generate random input tensors
# tgt_tf = tf.random.normal((sequence_length, batch_size, d_model))  # TensorFlow target
# memory_tf = tf.random.normal((sequence_length, batch_size, d_model))  # TensorFlow memory

# tgt_torch = torch.tensor(tgt_tf.numpy())  # PyTorch target
# memory_torch = torch.tensor( memory_tf.numpy() )  # PyTorch memory

# # Masks (optional)
# tgt_mask_tf = None  # TensorFlow mask
# memory_mask_tf = None  # TensorFlow mask

# tgt_mask_torch = None  # PyTorch mask
# memory_mask_torch = None  # PyTorch mask

# from util.torch_to_tf import nn_TransformerDecoder
# # Define TensorFlow model
# tf_decoder = nn_TransformerDecoder(n_layers, d_model, nhead, dim_feedforward, dropout, activation)

# # Call the TensorFlow model
# tf_output = tf_decoder(tgt_tf, memory_tf, tgt_mask=tgt_mask_tf, memory_mask=memory_mask_tf, training=True)

# print("\nTensorFlow Output:")
# print(tf_output)



# # Define PyTorch model
# torch_decoder = torch.nn.TransformerDecoder(n_layers, d_model, nhead, dim_feedforward, dropout, activation)

# # Call the PyTorch model
# torch_output = torch_decoder(tgt_torch, memory_torch, tgt_mask=tgt_mask_torch, memory_mask=memory_mask_torch)

# print("\nPyTorch Output:")
# print(torch_output)



# # Convert PyTorch output to NumPy for comparison
# torch_output_np = torch_output.detach().numpy()

# # Check if the outputs are close
# print("\nAre the outputs close?")
# print(np.allclose(tf_output.numpy(), torch_output_np, atol=1e-5))  # Check with a tolerance







import torch
import torch.nn as nn
import torch.nn.functional as F

# set random seeds to ensure the reproducibility
torch.manual_seed(42)

# create a TransformerDecoderLayer
d_model = 4
nhead = 2
decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=16, dropout=0)

# initialize weights manually to make output predictable**
# for param in decoder_layer.parameters():
#     param.data.fill_(0.5)

# input data with batch=1
tgt = torch.tensor([[[0.1, 0.2, 0.3, 0.4]],
                    [[0.5, 0.6, 0.7, 0.8]]])  # (tgt_len=2, batch_size=1, d_model=4)

memory = torch.tensor([[[0.9, 1.0, 1.1, 1.2]],
                       [[1.3, 1.4, 1.5, 1.6]]])  # (memory_len=2, batch_size=1, d_model=4)

# forward pass
output = decoder_layer(tgt, memory)

# print output
print("TransformerDecoderLayer output: \n", output)



from util.torch_to_tf import nn_TransformerDecoderLayer, torch_tensor

# create a TransformerDecoderLayer
d_model = 4
nhead = 2
decoder_layer = nn_TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=16, dropout=0)

# initialize weights manually to make the output predictable
# for param in decoder_layer.trainble_variables:
#     param.assign(0.5)


# input data with batch=1
tgt = torch_tensor( np.array([[[0.1, 0.2, 0.3, 0.4]],
                    [[0.5, 0.6, 0.7, 0.8]]]) )  # (tgt_len=2, batch_size=1, d_model=4)

memory = torch_tensor( np.array([[[0.9, 1.0, 1.1, 1.2]],
                       [[1.3, 1.4, 1.5, 1.6]]]) )  # (memory_len=2, batch_size=1, d_model=4)

# forward pass
output = decoder_layer(tgt, memory)

# print output
print("TransformerDecoderLayer output: \n", output)

