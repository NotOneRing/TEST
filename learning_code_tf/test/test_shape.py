
import tensorflow as tf

import torch

import numpy as np

arr = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])

# arr = arr.reshape(-1)

torch_tensor = torch.tensor(arr)

torch_tensor = torch_tensor.reshape(-1)

print("torch_tensor = ", torch_tensor)
print("torch_tensor.shape = ", torch_tensor.shape)

tf_tensor = tf.convert_to_tensor(arr)

tf_tensor = tf.reshape(tf_tensor, -1)

print("tf_tensor = ", tf_tensor)
print("tf_tensor.shape = ", tf_tensor.shape)

























