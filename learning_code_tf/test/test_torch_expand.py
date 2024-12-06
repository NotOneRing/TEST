
import torch

# 创建一个 1D Tensor
tensor = torch.tensor([1, 2, 3])

# 扩展为 3x3 的矩阵
expanded_tensor = tensor.expand(3, 3)

print(expanded_tensor)


import tensorflow as tf

# 创建一个 1D Tensor
tensor = tf.constant([1, 2, 3])

# 扩展为 3x3 的矩阵
expanded_tensor = tf.broadcast_to(tensor, [3, 3])

print(expanded_tensor)









