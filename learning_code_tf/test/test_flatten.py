import tensorflow as tf

import torch

# feats = 



# feats = feats.flatten(1, 2)

# feats = tf.reshape(feats, [feats.shape[0], -1])


import numpy as np
import tensorflow as tf
import torch

# Step 1: Create a NumPy array for feats
batch_size = 10
height = 6
width = 6
channels = 128
feats = np.random.rand(batch_size, height, width, channels).astype(np.float32)

# Step 2: Convert to TensorFlow tensor
feats_tf = tf.convert_to_tensor(feats)

# Step 3: Convert to PyTorch tensor
feats_torch = torch.tensor(feats)

# Step 4: Flatten the second and third dimensions (height and width) for each format
# NumPy
feats_np_flatten = feats.reshape(feats.shape[0], -1, feats.shape[-1])

# TensorFlow
feats_tf_flatten = tf.reshape(feats_tf, [feats_tf.shape[0], -1, feats_tf.shape[-1]])

# PyTorch
feats_torch_flatten = feats_torch.flatten(start_dim=1, end_dim=2)

# Step 5: Compare the results
# Convert TensorFlow and PyTorch tensors back to NumPy arrays for comparison
feats_tf_flatten_np = feats_tf_flatten.numpy()
feats_torch_flatten_np = feats_torch_flatten.numpy()

# Check if they are equal
print("NumPy vs TensorFlow:", np.allclose(feats_np_flatten, feats_tf_flatten_np))
print("NumPy vs PyTorch:", np.allclose(feats_np_flatten, feats_torch_flatten_np))
print("TensorFlow vs PyTorch:", np.allclose(feats_tf_flatten_np, feats_torch_flatten_np))

# Print the shape for verification
print("Shape after flattening:")
print("NumPy:", feats_np_flatten.shape)
print("TensorFlow:", feats_tf_flatten.shape)
print("PyTorch:", feats_torch_flatten.shape)
