
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

result = tensor.view(3, 3)

print(result)
print(result.shape)

result = tensor.view(3, 3, 1)


print("result = ", result)
print("result.shape = ", result.shape)

print("tensor = ", tensor)
print("tensor.shape = ", tensor.shape)



result = tensor.reshape(3, 3, 1)


print("result = ", result)
print("result.shape = ", result.shape)

print("tensor = ", tensor)
print("tensor.shape = ", tensor.shape)


import numpy as np

import tensorflow as tf

from util.torch_to_tf import torch_tensor_view


tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

tensor = tf.convert_to_tensor(tensor)


result = torch_tensor_view(tensor, 3, 3)
print(result)



result = torch_tensor_view(tensor, 1, 3, 3)
print(result)




