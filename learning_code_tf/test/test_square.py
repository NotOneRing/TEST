import torch

sigma = torch.tensor([1.0, 2.0, 3.0])
var = sigma**2  # 逐元素平方
print(var)  # 输出: tensor([1., 4., 9.])


import tensorflow as tf

sigma = tf.constant([1.0, 2.0, 3.0])
var = tf.square(sigma)  # 逐元素平方
print(var)  # 输出: [1. 4. 9.]


