import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np

from util.torch_to_tf import nn_Conv2d




def test1():
    # PyTorch Conv2d
    torch_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, bias=True)
    torch_conv.weight.data = torch.nn.init.kaiming_uniform_(torch_conv.weight, nonlinearity='relu')

    # TensorFlow Conv2d
    tf_conv = nn_Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, bias=True)
    # tf_conv.build((None, 3, 64, 64))  # 假设输入尺寸为 (N, C, H, W) = (None, 3, 64, 64)


    tf_input = tf.random.uniform([1, 3, 64, 64])
    _ = tf_conv(tf_input).numpy()

    # 设置相同权重
    torch_weights = torch_conv.weight.detach().numpy()  # PyTorch 权重
    # torch (out_channel, in_channel // groups, kernel_size[0], kernel_size[1])
    # tensorflow (kernel_size[0], kernel_size[1], in_channel // groups, out_channel)
    torch_weights_tf = np.transpose(torch_weights, (2, 3, 1, 0))  # 转换为 TensorFlow 形状
    tf_conv.conv2d.kernel.assign(torch_weights_tf)

    # 设置相同 bias
    if torch_conv.bias is not None:
        tf_conv.conv2d.bias.assign(torch_conv.bias.detach().numpy())


    print("PyTorch 权重:", torch_conv.weight.detach().numpy().shape)
    print("TensorFlow 权重:", tf_conv.conv2d.kernel.numpy().shape)


    # 生成相同输入
    torch_input = torch.randn(1, 3, 64, 64)  # PyTorch 输入
    tf_input = tf.convert_to_tensor(torch_input.numpy(), dtype=tf.float32)  # 转换为 TensorFlow 格式

    # 计算输出
    torch_output = torch_conv(torch_input).detach().numpy()
    tf_output = tf_conv(tf_input).numpy()

    # 计算误差
    diff = np.abs(torch_output - tf_output).max()
    print("max diff = :", diff)  # 应该接近 0


    assert np.allclose(torch_output , tf_output, atol=1e-5)




def test2():
    # PyTorch Conv2d
    torch_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, bias=False)
    torch_conv.weight.data = torch.nn.init.kaiming_uniform_(torch_conv.weight, nonlinearity='relu')

    # TensorFlow Conv2d
    tf_conv = nn_Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, bias=False)
    # tf_conv.build((None, 3, 64, 64))  # 假设输入尺寸为 (N, C, H, W) = (None, 3, 64, 64)


    tf_input = tf.random.uniform([1, 3, 64, 64])
    _ = tf_conv(tf_input).numpy()

    # 设置相同权重
    torch_weights = torch_conv.weight.detach().numpy()  # PyTorch 权重
    torch_weights_tf = np.transpose(torch_weights, (2, 3, 1, 0))  # 转换为 TensorFlow 形状
    tf_conv.conv2d.kernel.assign(torch_weights_tf)

    # # 设置相同 bias
    # if torch_conv.bias is not None:
    #     tf_conv.conv2d.bias.assign(torch_conv.bias.detach().numpy())


    print("PyTorch 权重:", torch_conv.weight.detach().numpy().shape)
    print("TensorFlow 权重:", tf_conv.conv2d.kernel.numpy().shape)


    # 生成相同输入
    torch_input = torch.randn(1, 3, 64, 64)  # PyTorch 输入
    tf_input = tf.convert_to_tensor(torch_input.numpy(), dtype=tf.float32)  # 转换为 TensorFlow 格式

    # 计算输出
    torch_output = torch_conv(torch_input).detach().numpy()
    tf_output = tf_conv(tf_input).numpy()

    # 计算误差
    diff = np.abs(torch_output - tf_output).max()
    print("max diff = :", diff)  # 应该接近 0


    assert np.allclose(torch_output , tf_output, atol=1e-5)





test1()

test2()







