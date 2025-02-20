import torch


import tensorflow as tf

from util.torch_to_tf import torch_prod

def test_prod():
    # 创建一个张量
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

    result0 = torch.prod(tensor)
    print(result0)

    # 沿着维度0计算乘积
    result1 = torch.prod(tensor, dim=0)
    print(result1)

    # 沿着维度1计算乘积
    result2 = torch.prod(tensor, dim=1)
    print(result2)


    # 创建一个张量
    tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

    result0 = torch_prod(tensor)
    print(result0)

    # 沿着维度0计算乘积
    result1 = torch_prod(tensor, dim=0)
    print(result1)

    # 沿着维度1计算乘积
    result2 = torch_prod(tensor, dim=1)
    print(result2)

test_prod()



