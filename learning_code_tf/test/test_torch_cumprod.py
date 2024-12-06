import torch

# 创建一个一维张量
tensor = torch.tensor([1, 2, 3, 4])

# 计算累积乘积
cumprod_result = torch.cumprod(tensor, dim=0)
print(cumprod_result)



tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 沿着维度 0（行）计算累积乘积
cumprod_dim_0 = torch.cumprod(tensor_2d, dim=0)
print("cumprod along dim 0:")
print(cumprod_dim_0)

# 沿着维度 1（列）计算累积乘积
cumprod_dim_1 = torch.cumprod(tensor_2d, dim=1)
print("cumprod along dim 1:")
print(cumprod_dim_1)



import tensorflow as tf

# 创建一个一维张量
tensor = tf.constant([1, 2, 3, 4])

# 计算累积乘积
cumprod_result = tf.math.cumprod(tensor, axis=0)
print(cumprod_result)



tensor_2d = tf.constant([[1, 2, 3], [4, 5, 6]])

# 沿着维度 0（行）计算累积乘积
cumprod_dim_0 = tf.math.cumprod(tensor_2d, axis=0)
print("cumprod along axis 0:")
print(cumprod_dim_0)

# 沿着维度 1（列）计算累积乘积
cumprod_dim_1 = tf.math.cumprod(tensor_2d, axis=1)
print("cumprod along axis 1:")
print(cumprod_dim_1)
