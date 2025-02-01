


import tensorflow as tf
import numpy as np

class nn_Conv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(nn_Conv2d, self).__init__()
        
        # 解析参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        print("self.use_bias = ", self.use_bias)

# torch_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
# tf_conv = tf.keras.layers.Conv2D(
#     filters=32, kernel_size=3, strides=1, padding="same", use_bias=True
# )
        # TensorFlow 的 Conv2D 需要 NHWC 格式，因此 filters 需要是 (H, W, C_in, C_out)
        self.conv2d = tf.keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="valid",  # 这里先不填充，我们自己手动填充
            # padding="same",  # 这里先不填充，我们自己手动填充
            dilation_rate=self.dilation,
            use_bias=self.use_bias,
            kernel_initializer="glorot_uniform",  # PyTorch 默认 kaiming_uniform
            # bias_initializer="zeros" if self.use_bias else None
            bias_initializer="glorot_uniform" if self.use_bias else None
        )

    def build(self, input_shape):
        """在这里调整 TensorFlow 的权重格式，使其匹配 PyTorch"""
        # PyTorch 的权重是 (C_out, C_in, H, W)，TensorFlow 是 (H, W, C_in, C_out)
        self.conv2d.build((None, None, None, self.in_channels))
        
        # 交换权重维度，使其匹配 PyTorch
        kernel = self.conv2d.kernel  # (H, W, C_in, C_out)
        kernel = tf.Variable(tf.transpose(kernel, [3, 2, 0, 1]), trainable=True)  # (C_out, C_in, H, W)
        self.conv2d.kernel = tf.Variable(tf.transpose(kernel, [2, 3, 1, 0]), trainable=True)  # (H, W, C_in, C_out)
        # print("input_shape = ", input_shape)

        # _ = self.conv2d( tf.random.uniform(input_shape) )


    def call(self, x):
        # PyTorch 输入是 (N, C, H, W)，转换成 TensorFlow 需要的 (N, H, W, C)
        x = tf.transpose(x, [0, 2, 3, 1])

        # 手动填充，模拟 PyTorch 的 padding
        if self.padding > 0:
            x = tf.pad(x, [
                [0, 0],  # batch 维度不填充
                [self.padding, self.padding],  # H 维度
                [self.padding, self.padding],  # W 维度
                [0, 0]  # C 维度不填充
            ])

        # 进行卷积
        x = self.conv2d(x)

        # 输出转换回 PyTorch 格式 (N, C, H, W)
        x = tf.transpose(x, [0, 3, 1, 2])
        return x

    def set_weights(self, weights):
        self.conv2d.set_weights(weights)








import torch
import torch.nn as nn

# PyTorch 版本
torch_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
torch_input = torch.randn(2, 3, 32, 32)  # NCHW
torch_output = torch_conv(torch_input)
print("PyTorch Output Shape:", torch_output.shape)  # (2, 32, 32, 32)

# TensorFlow 版本
tf_input = tf.convert_to_tensor(torch_input.numpy(), dtype=tf.float32)  # PyTorch -> TensorFlow
tf_conv = nn_Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
tf_output = tf_conv(tf_input)
print("TensorFlow Output Shape:", tf_output.shape)  # (2, 32, 32, 32)











import numpy as np
import torch
import tensorflow as tf

# 创建相同的 NumPy 输入数据
np_input = np.random.randn(2, 3, 32, 32).astype(np.float32)  # (N, C, H, W)

# PyTorch 格式 (N, C, H, W)
torch_input = torch.tensor(np_input, dtype=torch.float32)

# TensorFlow 格式 (N, H, W, C)
tf_input = tf.convert_to_tensor(np.transpose(np_input, (0, 2, 3, 1)), dtype=tf.float32)


import torch.nn as nn

# PyTorch Conv2D
torch_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)

# 获取 PyTorch 权重 & 偏置
torch_weights = torch_conv.weight.detach().numpy()  # (32, 3, 3, 3)
torch_bias = torch_conv.bias.detach().numpy()       # (32,)

# **转换 PyTorch 权重到 TensorFlow 格式**
tf_weights = np.transpose(torch_weights, (2, 3, 1, 0))  # (3, 3, 3, 32)

# **TensorFlow Conv2D**
# tf_conv = tf.keras.layers.Conv2D(
#     filters=32, kernel_size=3, strides=1, padding="same", use_bias=True
# )

tf_conv = nn_Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
# tf_conv = tf.keras.layers.Conv2D(
#     filters=32, kernel_size=3, strides=1, padding="same", use_bias=True
# )

# 构建 TensorFlow 层（触发权重初始化）
tf_conv.build((None, 32, 32, 3))

# **手动设置 TensorFlow 权重**
tf_conv.set_weights([tf_weights, torch_bias])


torch_output = torch_conv(torch_input)  # (N, 32, H, W)
torch_output_np = torch_output.detach().numpy()

tf_output = tf_conv(tf_input)  # (N, H, W, 32)
tf_output_np = tf_output.numpy()

# **转换回 (N, C, H, W) 以匹配 PyTorch**
tf_output_np = np.transpose(tf_output_np, (0, 3, 1, 2))


# 计算最大误差
max_diff = np.max(np.abs(torch_output_np - tf_output_np))
print("Max difference:", max_diff)

# 误差应该非常小（< 1e-5）
assert max_diff < 1e-5, "Outputs do not match!"


























