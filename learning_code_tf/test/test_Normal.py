import torch
from torch.distributions import Normal

# 定义均值和标准差
mean = torch.tensor([0.0, 1.0])
std = torch.tensor([1.0, 0.5])

# 创建正态分布对象
dist = Normal(mean, std)

# 采样
sample = dist.sample()  # 生成一个样本
print(f"Sample: {sample}")

# 计算概率密度函数值
log_prob = dist.log_prob(torch.tensor([1.0, 1.0]))  # 对应均值的概率密度
print(f"Log probability: {log_prob}")





import tensorflow as tf

import numpy as np

from util.torch_to_tf import Normal

def log_normal_pdf(x, mean, std):
    """
    计算正态分布的对数概率密度函数

    Args:
        x: 需要计算概率密度的点
        mean: 正态分布的均值
        std: 正态分布的标准差

    Returns:
        对数概率密度
    """
    # 计算PDF的对数
    log_pdf = -tf.math.log(std * tf.math.sqrt(2 * tf.constant(np.pi))) - 0.5 * ((x - mean) ** 2) / (std ** 2)
    return log_pdf

# 测试
mu = tf.convert_to_tensor(mean.numpy(), dtype=np.float32)  # 均值
sigma = tf.convert_to_tensor(std.numpy(), dtype=np.float32)  # 标准差

# 假设我们要计算 x = 1 的对数概率密度
x = tf.convert_to_tensor( np.array([1.0, 1.0]), dtype=np.float32 )

dist = Normal(mu, sigma)

# 采样
sample = dist.sample()  # 生成一个样本
print(f"Sample: {sample}")


log_prob = dist.log_prob(x)  # 对应均值的概率密度

# 计算
# log_prob = log_normal_pdf(x, mu, sigma)
print(f"Log probability at x = {x}: {log_prob.numpy()}")


print("log_prob = ", log_prob)

print("type(log_prob) = ", log_prob)



