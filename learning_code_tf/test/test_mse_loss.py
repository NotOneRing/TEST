import torch
import torch.nn.functional as F

# 示例数据
input = torch.tensor([0.5, 1.0, 1.5])
target = torch.tensor([1.0, 1.0, 1.0])

# 计算 MSE 损失
loss = F.mse_loss(input, target, reduction='mean')
print(loss)  # 输出均方误差损失

print( type(loss) )  

import tensorflow as tf

from util.torch_to_tf import torch_mse_loss

# 示例数据
input = tf.constant([0.5, 1.0, 1.5])
target = tf.constant([1.0, 1.0, 1.0])

# 计算 MSE 损失
loss = torch_mse_loss(input, target, reduction='mean')
print(loss)  # 输出均方误差损失

print( type(loss) )




