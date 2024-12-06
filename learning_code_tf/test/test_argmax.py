
import torch

# 示例张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 沿着第 0 维（行）计算最大值的索引
result = torch.argmax(tensor, dim=0)
print(result)
# 输出：tensor([1, 1, 1])  # 沿第0维每列的最大值索引

# 沿着第 1 维（列）计算最大值的索引
result = torch.argmax(tensor, dim=1)
print(result)
# 输出：tensor([2, 2])  # 沿第1维每行的最大值索引

import numpy as np

import tensorflow as tf

from util.torch_to_tf import torch_argmax

# 示例张量
tensor = np.array([[1, 2, 3], [4, 5, 6]])

tensor = tf.convert_to_tensor(tensor)

# 沿着第 0 维（行）计算最大值的索引
result = torch_argmax(tensor, dim=0)
print(result)
# 输出：tensor([1, 1, 1])  # 沿第0维每列的最大值索引

# 沿着第 1 维（列）计算最大值的索引
result = torch_argmax(tensor, dim=1)
print(result)
# 输出：tensor([2, 2])  # 沿第1维每行的最大值索引



