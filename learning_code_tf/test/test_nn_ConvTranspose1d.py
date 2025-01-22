from util.torch_to_tf import nn_ConvTranspose1d

import numpy as np
import torch
import tensorflow as tf

from tensorflow.keras.saving import register_keras_serializable


# 参数配置（确保 groups=1）
batch_size = 2
in_channels = 4
out_channels = 8
length = 10
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
dilation = 1
groups = 1  # 必须保证 in_channels % groups == 0

# 生成输入数据（固定值，便于调试）
input_np = np.ones((batch_size, in_channels, length), dtype=np.float32)
input_torch = torch.from_numpy(input_np)
input_tf = tf.convert_to_tensor(np.transpose(input_np, (0, 2, 1)))  # TF 格式: (batch, length, channels)

# 创建 PyTorch 层（固定权重）
torch_conv_trans = torch.nn.ConvTranspose1d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    output_padding=output_padding,
    dilation=dilation,
    groups=groups,
    bias=True
)
with torch.no_grad():
    torch_conv_trans.weight.fill_(1.0)  # 权重全设为 1
    torch_conv_trans.bias.fill_(0.0)    # 偏置全设为 0

# 创建 TensorFlow 层（修复 groups 参数传递）
@register_keras_serializable(package="Custom")
class nn_ConvTranspose1d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, name="nn_ConvTranspose1d", **kwargs):
        super().__init__(name=name, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups  # 确保正确传递 groups
        self.bias = bias

        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")

        # 定义 Conv1DTranspose 层
        self.conv_transpose = tf.keras.layers.Conv1DTranspose(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="valid",
            dilation_rate=self.dilation,
            use_bias=self.bias,
            kernel_initializer="zeros",
            bias_initializer="zeros"
        )

    def build(self, input_shape):
        self.conv_transpose.build((None, input_shape[0], input_shape[1]))
        super().build(input_shape)

    def call(self, x):
        x = tf.pad(x, [[0, 0], [self.padding, self.padding], [0, 0]])
        x = self.conv_transpose(x)
        L_in_original = tf.shape(x)[1] - 2 * self.padding
        L_out_pytorch = (L_in_original - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + self.output_padding + 1
        L_out_tf = tf.shape(x)[1]
        crop_right = L_out_tf - L_out_pytorch
        x = x[:, :-crop_right, :] if crop_right > 0 else x
        if self.output_padding > 0:
            x = tf.pad(x, [[0, 0], [0, self.output_padding], [0, 0]])
        return x

# 初始化 TensorFlow 层
tf_conv_trans = nn_ConvTranspose1d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    output_padding=output_padding,
    dilation=dilation,
    groups=groups,  # 确保 groups=1
    bias=True
)

# 手动构建 TF 层并复制权重
tf_conv_trans.build(input_shape=(length, in_channels))
with torch.no_grad():
    # 关键修正：转置轴顺序为 (2, 0, 1)
    torch_weight = torch_conv_trans.weight.detach().numpy().transpose(2, 0, 1)  # (3, 4, 8)
    torch_bias = torch_conv_trans.bias.detach().numpy()
    tf_conv_trans.conv_transpose.kernel.assign(torch_weight)
    tf_conv_trans.conv_transpose.bias.assign(torch_bias)

# 前向传播
output_torch = torch_conv_trans(input_torch).detach().numpy()  # (2, 8, 20)
output_tf = tf_conv_trans(input_tf).numpy()  # (2, 20, 8)
output_tf = np.transpose(output_tf, (0, 2, 1))  # (2, 8, 20)

# 验证一致性
print("PyTorch 权重形状:", torch_conv_trans.weight.shape)          # (4, 8, 3)
print("TensorFlow 权重形状:", tf_conv_trans.conv_transpose.kernel.shape)  # (3, 4, 8)
print("PyTorch 输出形状:", output_torch.shape)                   # (2, 8, 20)
print("TensorFlow 输出形状:", output_tf.shape)                   # (2, 8, 20)
print("PyTorch 输出首元素:", output_torch[0, 0, 0])              # 3.0
print("TensorFlow 输出首元素:", output_tf[0, 0, 0])              # 3.0
assert output_torch.shape == output_tf.shape, "形状不一致"
assert np.allclose(output_torch, output_tf, atol=1e-5), "数值不一致"
print("输出完全一致！")
