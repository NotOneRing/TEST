import numpy as np
import torch
import tensorflow as tf
from util.torch_to_tf import nn_Conv1d  # 假设 nn_Conv1d 实现保存在 nn_Conv1d.py 中

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# 生成随机输入数据 (batch_size, channels, length)
batch_size = 2
in_channels = 4
out_channels = 8
length = 10
kernel_size = 3
stride = 1
padding = 1
dilation = 1
groups = 1

# 生成 numpy 数据
input_np = np.random.randn(batch_size, in_channels, length).astype(np.float32)

# 将 numpy 数据转换为 PyTorch 张量
input_torch = torch.from_numpy(input_np)

# 将 numpy 数据转换为 TensorFlow 张量
input_tf = tf.convert_to_tensor(np.transpose(input_np, (0, 2, 1)))  # TensorFlow 需要 (batch_size, length, channels)

# 创建 PyTorch 的 Conv1d 层
torch_conv1d = torch.nn.Conv1d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    groups=groups,
    bias=True
)

# 创建 TensorFlow 的 Conv1d 层
tf_conv1d = nn_Conv1d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    groups=groups,
    bias=True
)

# 手动构建 TensorFlow 层（指定输入形状）
input_shape = (batch_size, length, in_channels)  # TensorFlow 输入形状为 (batch_size, length, channels)
tf_conv1d.build(input_shape=input_shape)

# 将 PyTorch 的权重和偏置复制到 TensorFlow 层
with torch.no_grad():
    # PyTorch 权重形状: (out_channels, in_channels // groups, kernel_size)
    # TensorFlow 权重形状: (kernel_size, in_channels // groups, out_channels)
    tf_kernel = torch_conv1d.weight.numpy().transpose(2, 1, 0)  # 调整维度顺序
    tf_bias = torch_conv1d.bias.numpy()

    # 通过 weights 列表访问 TensorFlow 层的权重和偏置
    tf_conv1d.conv1d.weights[0].assign(tf_kernel)
    tf_conv1d.conv1d.weights[1].assign(tf_bias)

# 前向传播
# PyTorch
output_torch = torch_conv1d(input_torch).detach().numpy()  # 形状: (batch_size, out_channels, length)

# TensorFlow
output_tf = tf_conv1d(input_tf).numpy()  # 形状: (batch_size, length, out_channels)
output_tf = np.transpose(output_tf, (0, 2, 1))  # 转换为 (batch_size, out_channels, length)

# 比较输出
print("PyTorch Output Shape:", output_torch.shape)
print("TensorFlow Output Shape:", output_tf.shape)

# 检查输出是否一致
if np.allclose(output_torch, output_tf, atol=1e-5):
    print("Outputs are consistent!")
else:
    print("Outputs are NOT consistent!")