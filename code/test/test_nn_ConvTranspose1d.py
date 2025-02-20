import tensorflow as tf

from util.torch_to_tf import nn_ConvTranspose1d


import torch
import numpy as np

# PyTorch ConvTranspose1d
torch_layer = torch.nn.ConvTranspose1d(3, 3, 4, 2, 1)  # padding=1
torch_input = torch.randn(1, 3, 10)
torch_output = torch_layer(torch_input).detach().numpy()


# TensorFlow ConvTranspose1d
tf_layer = nn_ConvTranspose1d(3, 3, 4, 2, 1)  # 只实现 padding=1
tf_input = tf.convert_to_tensor(torch_input.numpy(), dtype=tf.float32)
tf_output = tf_layer(tf_input).numpy()

# same weights
torch_weights = torch_layer.weight.detach().numpy()  # PyTorch 权重
torch_weights_tf = np.transpose(torch_weights, (2, 1, 0))  # 转换为 TensorFlow 形状
tf_layer.conv1d_transpose.kernel.assign(torch_weights_tf)

# same bias
if torch_layer.bias is not None:
    tf_layer.conv1d_transpose.bias.assign(torch_layer.bias.detach().numpy())

tf_output = tf_layer(tf_input).numpy()


print("最大误差:", np.abs(torch_output - tf_output).max())
print("PyTorch 输出形状:", torch_output.shape)
print("TensorFlow 输出形状:", tf_output.shape)


