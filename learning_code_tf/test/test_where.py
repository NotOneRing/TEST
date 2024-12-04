

import numpy as np
import torch
import tensorflow as tf

# 创建一个测试的 numpy 数组
np_array = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])

# 转换为 torch.tensor 和 tf.Tensor
torch_tensor = torch.tensor(np_array)
tf_tensor = tf.convert_to_tensor(np_array)

# 使用 where 函数：保留正值，负值替换为 0
torch_result = torch.where(torch_tensor > 0, torch_tensor, 0)
tf_result = tf.where(tf_tensor > 0, tf_tensor, 0)

# 输出结果
print("Torch Result:\n", torch_result.numpy())
print("TensorFlow Result:\n", tf_result.numpy())





