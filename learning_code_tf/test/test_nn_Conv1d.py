import numpy as np
import torch
import tensorflow as tf
from util.torch_to_tf import nn_Conv1d  # 假设 nn_Conv1d 实现保存在 nn_Conv1d.py 中


# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# 参数设置
batch_size = 2
in_channels = 4
out_channels = 8
length = 10
kernel_size = 3
stride = 1
padding = 1
dilation = 1
groups = 1  # 确保 in_channels % groups == 0

# 生成输入数据 (PyTorch 格式: batch_size, in_channels, length)
input_np = np.random.randn(batch_size, in_channels, length).astype(np.float32)
input_torch = torch.from_numpy(input_np)

# input_tf = tf.convert_to_tensor(np.transpose(input_np, (0, 2, 1)))  # 转换为 TF 格式: (batch, length, channels)
input_tf = tf.convert_to_tensor(input_np)  # 转换为 TF 格式: (batch, length, channels)


# 创建 PyTorch 层
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

# 创建 TensorFlow 层
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

# 手动构建 TF 层（输入形状不含 batch_size）
input_shape = (length, in_channels)  # 正确形状为 (length, in_channels)
tf_conv1d.build(input_shape=input_shape)

# 复制权重
with torch.no_grad():
    # 调整权重形状
    torch_weight = torch_conv1d.weight.numpy().transpose(2, 1, 0)  
    # pytorch is (out_channels, in_channels, kernel_size)
    # tensorflow is (kernel_size, in_channels, out_channels)
    torch_bias = torch_conv1d.bias.numpy()
    
    # 检查 TF 层的权重是否已初始化
    if len(tf_conv1d.conv1d.weights) == 0:
        raise RuntimeError("Conv1D 层未初始化，请检查 TensorFlow 版本或输入形状。")
    
    # 分配权重和偏置
    tf_conv1d.conv1d.kernel.assign(torch_weight)
    tf_conv1d.conv1d.bias.assign(torch_bias)

# 前向传播
output_torch = torch_conv1d(input_torch).detach().numpy()  # (batch, out_channels, length)
output_tf = tf_conv1d(input_tf).numpy()  
# (batch, length, out_channels)

# output_tf = np.transpose(output_tf, (0, 2, 1))  # 转换为 (batch, out_channels, length)

# 检查一致性
if np.allclose(output_torch, output_tf, atol=1e-5):
    print("Outputs are consistent!")
else:
    print("Outputs are NOT consistent!")



















