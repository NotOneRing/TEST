import numpy as np
import torch
import tensorflow as tf

from util.torch_to_tf import torch_from_numpy

def test_from_numpy():
        
    # 使用 NumPy 生成一个数组
    numpy_array = np.full((3, 3), 5.0)  # 创建一个3x3的数组，填充5.0

    # 将 NumPy 数组转换为 PyTorch tensor 和 TensorFlow tensor
    torch_tensor = torch.from_numpy(numpy_array)
    tf_tensor = torch_from_numpy(numpy_array)


    assert np.allclose(torch_tensor.numpy(), tf_tensor.numpy())

test_from_numpy()




