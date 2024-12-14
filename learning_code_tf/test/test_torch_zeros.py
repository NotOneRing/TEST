import torch
import tensorflow as tf
import numpy as np


from util.torch_to_tf import torch_zeros


# 测试函数，比较两个库中的零张量
def test_zeros_equivalence(shape):
    # TensorFlow 创建的零张量
    tf_tensor = torch_zeros(shape, dtype=tf.float32)

    # PyTorch 创建的零张量
    torch_tensor = torch.zeros(shape, dtype=torch.float32)

    # 比较 TensorFlow 和 PyTorch 输出的张量
    print(f"TensorFlow zeros tensor:\n{tf_tensor.numpy()}")
    print(f"PyTorch zeros tensor:\n{torch_tensor.numpy()}")

    # 检查它们的输出是否相等
    match = np.allclose(tf_tensor.numpy(), torch_tensor.numpy())
    print(f"Outputs match: {match}")




    # TensorFlow 创建的零张量
    tf_tensor = torch_zeros(*shape, dtype=tf.float32)

    # PyTorch 创建的零张量
    torch_tensor = torch.zeros(*shape, dtype=torch.float32)

    # 比较 TensorFlow 和 PyTorch 输出的张量
    print(f"TensorFlow zeros tensor:\n{tf_tensor.numpy()}")
    print(f"PyTorch zeros tensor:\n{torch_tensor.numpy()}")

    # 检查它们的输出是否相等
    match = np.allclose(tf_tensor.numpy(), torch_tensor.numpy())
    print(f"Outputs match: {match}")



    return match

# 运行测试
shape = (2, 3)  # 设置测试的形状

test_zeros_equivalence(shape)


