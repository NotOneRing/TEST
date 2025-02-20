

from util.torch_to_tf import nn_GroupNorm, torch_tensor_permute
import numpy as np
import torch
import tensorflow as tf


# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# 生成随机输入数据 (batch_size, height, width, channels)
# case1:
batch_size = 2
height = 4
width = 4
channels = 8
num_groups = 2

# case2:
batch_size = 256
channels = 64
height = 1
width = 4
num_groups = 8

# (256, 64, 1, 4)

# 生成 numpy 数据
input_np = np.random.randn(batch_size, channels, height, width).astype(np.float32)

# 将 numpy 数据转换为 PyTorch 张量
input_torch = torch.from_numpy(input_np)
# .permute(0, 3, 1, 2)  # (batch_size, height, width, channels) 转换为PyTorch 需要 (batch_size, channels, height, width)

# .permute(0, 2, 3, 1)
# (batch_size, channels, height, width) 转换为 (batch_size, height, width, channels)

# 将 numpy 数据转换为 TensorFlow 张量
# input_tf = torch_tensor_permute(tf.convert_to_tensor(input_np), [0, 2, 3, 1])  # TensorFlow 需要 (batch_size, height, width, channels)
input_tf = tf.convert_to_tensor(input_np)  # TensorFlow 需要 (batch_size, height, width, channels)

# 创建 PyTorch 的 GroupNorm 层
torch_group_norm = torch.nn.GroupNorm(num_groups, channels, eps=1e-5, affine=True)

# 创建 TensorFlow 的 GroupNorm 层
tf_group_norm = nn_GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-5, affine=True)



# 将 PyTorch 的 gamma 和 beta 参数复制到 TensorFlow 层
with torch.no_grad():
    tf_group_norm.gamma.assign(torch_group_norm.weight.numpy())
    tf_group_norm.beta.assign(torch_group_norm.bias.numpy())

# 前向传播
# PyTorch
output_torch = torch_group_norm(input_torch)
# .permute(0, 2, 3, 1)  # (batch_size, channels, height, width) 转换回 (batch_size, height, width, channels)
output_torch_np = output_torch.detach().numpy()

print("PyTorch Output:\n", output_torch_np)

# # TensorFlow
# output_tf_np = torch_tensor_permute(tf_group_norm(input_tf), [0,3,1,2]).numpy()

output_tf_np = tf_group_norm(input_tf).numpy()

# 比较输出
print("TensorFlow Output:\n", output_tf_np)

# 检查输出是否一致
if np.allclose(output_torch_np, output_tf_np, atol=1e-5):
    print("Outputs are consistent!")
else:
    print("Outputs are NOT consistent!")




























