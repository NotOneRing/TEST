import torch

# Define a tensor and wrap it in nn.Parameter
data_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
param_torch = torch.nn.Parameter(data_torch, requires_grad=True)

# Print param details
print("PyTorch Parameter:")
print(f"Data: {param_torch.data}")
print(f"Requires grad: {param_torch.requires_grad}")


import tensorflow as tf

from util.torch_to_tf import nn_Parameter

# Define the data and create a TensorFlow parameter
data = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
param = nn_Parameter(data, requires_grad=True)

# Print param details
print("\nTensorFlow Parameter:")
print(f"Data: {param.numpy()}")
print(f"Requires grad: {param.trainable}")



import numpy as np

# PyTorch check
torch_param_data = param_torch.data.numpy()
torch_requires_grad = param_torch.requires_grad

# TensorFlow check
tf_param_data = param.numpy()
tf_requires_grad = param.trainable

# Check if data is the same
print("\nChecking if the data is the same:")
print(np.allclose(torch_param_data, tf_param_data))  # Should print True if data is the same

# Check if requires_grad/trainable is the same
print("Checking if requires_grad/trainable is the same:")
print(torch_requires_grad == tf_requires_grad)  # Should print True if both are True






