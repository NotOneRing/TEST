

import torch

import numpy as np

import tensorflow as tf


x = np.array([[1, 2, 3],
                  [4, 0, 6],
                  [7, 8, 9]])
# 定义张量
torch_x = torch.tensor(x)

# 沿着第 0 维计算最小值
min_result = torch_x.min(dim=0)

print("min_result = ", min_result)

print("Values:", min_result.values)  # 最小值
print("Indices:", min_result.indices)  # 最小值的索引



min_result = torch_x.min()

print("min_result = ", min_result)

print("Values:", min_result.values)  # 最小值
print("Indices:", min_result.indices)  # 最小值的索引


from util.torch_to_tf import torch_min


tf_x = tf.convert_to_tensor(x)

# 沿着第 0 维计算最小值
min_result = torch_min(tf_x, dim=0)

print("min_result = ", min_result)

print("Values:", min_result.values)  # 最小值
print("Indices:", min_result.indices)  # 最小值的索引



min_result = torch_min(tf_x)

print("min_result = ", min_result)

# print("Values:", min_result.values)  # 最小值
# print("Indices:", min_result.indices)  # 最小值的索引







