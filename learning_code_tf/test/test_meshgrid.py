import torch

# 创建两个一维张量
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5])

# 使用 meshgrid 创建坐标网格
xx, yy = torch.meshgrid(x, y, indexing='ij')
# xx, yy = torch.meshgrid(x, y, indexing='xy')
# xx, yy = torch.meshgrid(x, y)

print("xx:")
print(xx)

print("xx.shape:")
print(xx.shape)

print("yy:")
print(yy)

print("yy.shape:")
print(yy.shape)






import tensorflow as tf

from util.torch_to_tf import torch_meshgrid

# 创建两个一维张量
x = tf.constant([1, 2, 3])
y = tf.constant([4, 5])

# 使用 meshgrid 创建坐标网格
# xx, yy = torch_meshgrid(x, y)
xx, yy = tf.meshgrid(x, y, indexing = "ij")
# xx, yy = tf.meshgrid(x, y, indexing = "xy")

print("xx:")
print(xx)

print("yy:")
print(yy)







































