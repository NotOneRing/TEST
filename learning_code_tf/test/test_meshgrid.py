import torch

import numpy as np

import tensorflow as tf

from util.torch_to_tf import torch_meshgrid


def test_meshgrid():
    # 创建两个一维张量
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5])


    print("x:")
    print(x)

    print("y:")
    print(y)


    # 使用 meshgrid 创建坐标网格
    xx1, yy1 = torch.meshgrid(x, y, indexing='ij')
    # xx, yy = torch.meshgrid(x, y, indexing='xy')
    # xx, yy = torch.meshgrid(x, y)

    print("xx:")
    print(xx1)

    print("xx.shape:")
    print(xx1.shape)

    print("yy:")
    print(yy1)

    print("yy.shape:")
    print(yy1.shape)


    # 使用 meshgrid 创建坐标网格
    xx2, yy2 = torch.meshgrid([x, y], indexing='ij')
    # xx, yy = torch.meshgrid(x, y, indexing='xy')
    # xx, yy = torch.meshgrid(x, y)

    print("xx:")
    print(xx2)

    print("xx.shape:")
    print(xx2.shape)

    print("yy:")
    print(yy2)

    print("yy.shape:")
    print(yy2.shape)





    # 创建两个一维张量
    x = tf.constant([1, 2, 3])
    y = tf.constant([4, 5])


    print("x:")
    print(x)

    print("y:")
    print(y)



    # 使用 meshgrid 创建坐标网格
    # xx, yy = torch_meshgrid(x, y)
    xx1_tf, yy1_tf = torch_meshgrid(x, y, indexing = "ij")
    # xx, yy = tf.meshgrid(x, y, indexing = "xy")

    print("xx:")
    print(xx1_tf)

    print("yy:")
    print(yy1_tf)





    # 使用 meshgrid 创建坐标网格
    # xx, yy = torch_meshgrid(x, y)
    xx2_tf, yy2_tf = torch_meshgrid([x, y], indexing = "ij")
    # xx, yy = tf.meshgrid(x, y, indexing = "xy")

    print("xx:")
    print(xx2_tf)

    print("yy:")
    print(yy2_tf)






    assert np.allclose(xx1.numpy(), xx1_tf.numpy())

    assert np.allclose(yy1.numpy(), yy1_tf.numpy())

    assert np.allclose(xx2.numpy(), xx2_tf.numpy())

    assert np.allclose(yy2.numpy(), yy2_tf.numpy())






test_meshgrid()



































