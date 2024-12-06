import torch

# 创建一个浮动类型的张量
tensor = torch.tensor([1.1, 2.5, 3.7, -1.4])

# 四舍五入
rounded_tensor = torch.round(tensor)

print(rounded_tensor)




import tensorflow as tf

# 创建一个浮动类型的张量
tensor = tf.constant([1.1, 2.5, 3.7, -1.4])

# 四舍五入
rounded_tensor = tf.round(tensor)

print(rounded_tensor)
