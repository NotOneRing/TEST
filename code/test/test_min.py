

import torch

import numpy as np

import tensorflow as tf


x = np.array([[1, 2, 3],
                  [4, 0, 6],
                  [7, 8, 9]])
# define the tensor
torch_x = torch.tensor(x)

# calculate the min value along dimension 0
min_result = torch_x.min(dim=0)

print("min_result = ", min_result)

print("Values:", min_result.values)  # the min value
print("Indices:", min_result.indices)  # the index of the min value



min_result = torch_x.min()

print("min_result = ", min_result)

print("Values:", min_result.values)  # the min value
print("Indices:", min_result.indices)  # the index of the min value


from util.torch_to_tf import torch_min


tf_x = tf.convert_to_tensor(x)

# calculate the min value along dimension 0
min_result = torch_min(tf_x, dim=0)

print("min_result = ", min_result)

print("Values:", min_result.values)  # the min value
print("Indices:", min_result.indices)  # the index of the min value



min_result = torch_min(tf_x)

print("min_result = ", min_result)

# print("Values:", min_result.values)  # the min value
# print("Indices:", min_result.indices)  # the index of the min value







