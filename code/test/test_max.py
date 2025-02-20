

import torch

import numpy as np

import tensorflow as tf


x = np.array([[1, 2, 3],
                  [4, 0, 6],
                  [7, 8, 9]])

torch_x = torch.tensor(x)


max_result = torch_x.max(dim=0)

print("max_result = ", max_result)

print("Values:", max_result.values)
print("Indices:", max_result.indices)



max_result = torch_x.max()

print("max_result = ", max_result)

print("Values:", max_result.values)
print("Indices:", max_result.indices)


from util.torch_to_tf import torch_max


tf_x = tf.convert_to_tensor(x)


max_result = torch_max(tf_x, dim=0)

print("max_result = ", max_result)

print("Values:", max_result.values)

print("Indices:", max_result.indices)



max_result = torch_max(tf_x)

print("max_result = ", max_result)







