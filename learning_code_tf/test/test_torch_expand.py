
import torch

# 创建一个 1D Tensor
tensor = torch.tensor([1, 2, 3])

# 扩展为 3x3 的矩阵
expanded_tensor = tensor.expand(3, 3)

print(expanded_tensor)

expanded_tensor = tensor.expand([3, 3])

print(expanded_tensor)

import tensorflow as tf

# 创建一个 1D Tensor
tensor = tf.constant([1, 2, 3])

from util.torch_to_tf import torch_tensor_expand

# 扩展为 3x3 的矩阵
expanded_tensor = torch_tensor_expand(tensor, [3, 3])

print(expanded_tensor)


expanded_tensor = torch_tensor_expand(tensor, 3, 3)

print(expanded_tensor)








