


import torch
from torch.distributions import Normal, Independent

# # 定义两个独立的正态分布
# normal1 = Normal(torch.zeros(5), torch.ones(5))  # 均值为0，标准差为1
# normal2 = Normal(torch.ones(5), torch.ones(5))  # 均值为1，标准差为1

# # 将它们包装成独立分布
# independent_distribution = Independent(normal1, reinterpreted_batch_ndims=1)

# # 采样
# samples = independent_distribution.sample()
# print(f"Independent Samples: {samples}")










# import torch
# from torch.distributions import Independent
# , Normal 

import tensorflow as tf

import numpy as np

from util.torch_to_tf import Normal, Independent, torch_zeros, torch_ones
# , Indepan

# 定义一个批次数据，每个数据点是两个独立的正态分布
base_distribution = Normal(torch_zeros(3, 2), torch_ones(3, 2))  # 3个数据点，每个点有2个独立正态分布

# 使用 Independent 将它们视为独立分布
independent_distribution = Independent(base_distribution, reinterpreted_batch_ndims=1)

# 采样
samples = independent_distribution.sample()
print(f"1Independent Samples (Batch): {samples}")


# x = tf.convert_to_tensor( np.array([1.0, 1.0, 2.0]), dtype=np.float32 )

x = samples

# 计算
log_prob = independent_distribution.log_prob(x)  # 对应均值的概率密度
print(f"1Log probability at x = {x}: {log_prob.numpy()}")















import torch

from torch.distributions import Normal, Independent


# 定义一个批次数据，每个数据点是两个独立的正态分布
base_distribution = Normal(torch.zeros(3, 2), torch.ones(3, 2))  # 3个数据点，每个点有2个独立正态分布

# 使用 Independent 将它们视为独立分布
independent_distribution = Independent(base_distribution, reinterpreted_batch_ndims=1)

# # 采样
samples = torch.tensor(samples.numpy())
print(f"2Independent Samples (Batch): {samples}")

# samples = independent_distribution.sample()
# print(f"2Independent Samples (Batch): {samples}")

# x = torch.tensor(x.numpy())
x = torch.tensor(samples)


log_prob = independent_distribution.log_prob(x)  

# 计算
# log_prob = log_normal_pdf(x, mu, sigma)
print(f"2Log probability at x = {x}: {log_prob.numpy()}")













# import torch
# from torch.distributions import Normal, Independent

# # 定义一个三维的正态分布
# base_distribution = Normal(torch.zeros(2, 3, 4), torch.ones(2, 3, 4))  # 形状为 (2, 3, 4)

# # 使用 Independent 处理批次中的独立分布
# independent_distribution = Independent(base_distribution, reinterpreted_batch_ndims=2)

# # 采样
# samples = independent_distribution.sample()
# print(f"Independent Samples (3D Batch): {samples}")































