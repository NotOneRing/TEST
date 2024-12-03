import numpy as np
import torch
import tensorflow as tf

# 使用 NumPy 生成一个数组
numpy_array = np.full((3, 3), 5.0)  # 创建一个3x3的数组，填充5.0

# 将 NumPy 数组转换为 PyTorch tensor 和 TensorFlow tensor
torch_tensor = torch.tensor(numpy_array, dtype=torch.float32)
tf_tensor = tf.convert_to_tensor(numpy_array, dtype=tf.float32)

# 使用 torch.full 创建一个与 NumPy 数组相同形状和值的 PyTorch tensor
torch_full = torch.full((3, 3), 5.0, dtype=torch.float32)

# 使用 tf.fill 创建一个与 NumPy 数组相同形状和值的 TensorFlow tensor
tf_fill = tf.fill([3, 3], 5.0)

# 打印结果
print("Original NumPy array:\n", numpy_array)
print("\nTorch tensor:\n", torch_tensor)
print("\nTensorFlow tensor:\n", tf_tensor)
print("\nTorch full tensor:\n", torch_full)
print("\nTensorFlow fill tensor:\n", tf_fill)
