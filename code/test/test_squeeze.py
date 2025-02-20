import tensorflow as tf
import torch

import numpy as np

def test_squeeze():

    # 创建一个张量，包含维度为1的维度
    x_torch = torch.tensor([[[1, 2, 3], [4, 5, 6]]])  # 形状是 (1, 2, 3)

    # 使用 torch.squeeze 去除维度为1的维度
    x_torch_squeezed = torch.squeeze(x_torch)
    print("PyTorch Squeeze Output:")
    print(x_torch_squeezed)
    print("Shape after squeeze:", x_torch_squeezed.shape)  # 输出 (2, 3)



    # 创建一个张量，包含维度为1的维度
    x_tf = tf.constant([[[1, 2, 3], [4, 5, 6]]])  # 形状是 (1, 2, 3)

    # 使用 tf.squeeze 去除维度为1的维度
    x_tf_squeezed = tf.squeeze(x_tf)
    print("TensorFlow Squeeze Output:")
    print(x_tf_squeezed)
    print("Shape after squeeze:", x_tf_squeezed.shape)  # 输出 (2, 3)


    assert np.allclose(x_torch_squeezed.numpy(), x_tf_squeezed.numpy())

    # 创建一个张量，包含维度为1的维度
    x_torch = torch.tensor([[[1, 2, 3], [4, 5, 6]]])  # 形状是 (1, 2, 3)

    # 使用 torch.squeeze，指定去除维度 0 (即第一个维度) 为 1 的维度
    x_torch_squeezed = torch.squeeze(x_torch, dim=0)
    print("PyTorch Squeeze Output (dim=0):")
    print(x_torch_squeezed)
    print("Shape after squeeze:", x_torch_squeezed.shape)  # 输出 (2, 3)



    # 创建一个张量，包含维度为1的维度
    x_tf = tf.constant([[[1, 2, 3], [4, 5, 6]]])  # 形状是 (1, 2, 3)

    # 使用 tf.squeeze，指定去除维度 0 (即第一个维度) 为 1 的维度
    x_tf_squeezed = tf.squeeze(x_tf, axis=0)
    print("TensorFlow Squeeze Output (axis=0):")
    print(x_tf_squeezed)
    print("Shape after squeeze:", x_tf_squeezed.shape)  # 输出 (2, 3)

    assert np.allclose(x_torch_squeezed.numpy(), x_tf_squeezed.numpy())


test_squeeze()


