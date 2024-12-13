import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_unravel_index

# Input data
shape = (4, 3, 2)
linear_indices = [0, 5, 11, 17, 23]

# PyTorch test
pytorch_indices = torch.tensor(linear_indices)
pytorch_shape = shape
pytorch_result = torch.unravel_index(pytorch_indices, pytorch_shape)
print("PyTorch result:")
print([x.numpy() for x in pytorch_result])

# TensorFlow test
tf_indices = tf.constant(linear_indices, dtype=tf.int32)
tf_shape = shape
tf_result = torch_unravel_index(tf_indices, tf_shape)
print("\nTensorFlow result:")
print([x.numpy() for x in tf_result])

# Compare results
results_match = all(np.array_equal(torch_res.numpy(), tf_res.numpy()) for torch_res, tf_res in zip(pytorch_result, tf_result))
if results_match:
    print("\nThe results match!")
else:
    print("\nThe results do not match!")










