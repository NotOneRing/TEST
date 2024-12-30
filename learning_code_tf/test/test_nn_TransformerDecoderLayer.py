import numpy as np
import tensorflow as tf
import torch

# Parameters
d_model = 8
nhead = 2
dim_feedforward = 16
dropout = 0.1
activation = "relu"
n_layers = 2
sequence_length = 5
batch_size = 3

# Generate random input tensors
tgt_tf = tf.random.normal((sequence_length, batch_size, d_model))  # TensorFlow target
memory_tf = tf.random.normal((sequence_length, batch_size, d_model))  # TensorFlow memory

tgt_torch = torch.tensor(tgt_tf.numpy())  # PyTorch target
memory_torch = torch.tensor( memory_tf.numpy() )  # PyTorch memory

# Masks (optional)
tgt_mask_tf = None  # TensorFlow mask
memory_mask_tf = None  # TensorFlow mask

tgt_mask_torch = None  # PyTorch mask
memory_mask_torch = None  # PyTorch mask

from util.torch_to_tf import nn_TransformerDecoder
# Define TensorFlow model
tf_decoder = nn_TransformerDecoder(n_layers, d_model, nhead, dim_feedforward, dropout, activation)

# Call the TensorFlow model
tf_output = tf_decoder(tgt_tf, memory_tf, tgt_mask=tgt_mask_tf, memory_mask=memory_mask_tf, training=True)

print("\nTensorFlow Output:")
print(tf_output)



# Define PyTorch model
torch_decoder = nn_TransformerDecoder(n_layers, d_model, nhead, dim_feedforward, dropout, activation)

# Call the PyTorch model
torch_output = torch_decoder(tgt_torch, memory_torch, tgt_mask=tgt_mask_torch, memory_mask=memory_mask_torch)

print("\nPyTorch Output:")
print(torch_output)



# Convert PyTorch output to NumPy for comparison
torch_output_np = torch_output.detach().numpy()

# Check if the outputs are close
print("\nAre the outputs close?")
print(np.allclose(tf_output.numpy(), torch_output_np, atol=1e-5))  # Check with a tolerance







