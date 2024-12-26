

import numpy as np
import torch
import tensorflow as tf

from util.torch_to_tf import torch_where


def test_where():
    # 创建一个测试的 numpy 数组
    np_array = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])

    # 转换为 torch.tensor 和 tf.Tensor
    torch_tensor = torch.tensor(np_array)
    tf_tensor = tf.convert_to_tensor(np_array)

    # 使用 where 函数：保留正值，负值替换为 0
    torch_result = torch.where(torch_tensor > 0, torch_tensor, 0)

    tf_result = torch_where(tf_tensor > 0, tf_tensor, 0)

    # 输出结果
    print("Torch Result:\n", torch_result.numpy())
    print("TensorFlow Result:\n", tf_result.numpy())



    tensor = torch.tensor(
    [False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False])

    torch_result1 = torch.where(tensor)
    # [0]

    print("torch_result1 = ", torch_result1)


    tf_tensor = tf.convert_to_tensor(np.array(
    [False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False]))

    tf_result1 = torch_where(tensor)
    # [0]

    print("tf_result1 = ", tf_result1)


    # print("np.allclose(torch_result1.numpy(), tf_result1.numpy()) = ", np.allclose(torch_result1[0].numpy(), tf_result1.numpy()))
    print("np.allclose(torch_result1.numpy(), tf_result1.numpy()) = ", np.allclose(torch_result1[0].numpy(), tf_result1[0].numpy()))
    assert np.allclose(torch_result1[0].numpy(), tf_result1[0].numpy())

    tensor = torch.tensor(
    [True, False, True, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False])

    torch_result2 = torch.where(tensor)
    # [0]

    print("torch_result2 = ", torch_result2)

    print("torch_result2[0].shape = ", torch_result2[0].shape)


    tf_tensor = tf.convert_to_tensor(np.array(
    [True, False, True, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False]))

    tf_result2 = torch_where(tensor)
    # [0]

    print("tf_result2 = ", tf_result2)



    # print("np.allclose(torch_result2.numpy(), tf_result2.numpy()) = ", np.allclose(torch_result2[0].numpy(), tf_result2.numpy()))
    print("np.allclose(torch_result2.numpy(), tf_result2.numpy()) = ", np.allclose(torch_result2[0].numpy(), tf_result2[0].numpy()))
    assert np.allclose(torch_result2[0].numpy(), tf_result2[0].numpy())




    tensor = torch.tensor(
    [True, False, True, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False]).reshape(2, 2, 10)

    torch_result3 = torch.where(tensor)
    # [0]

    print("torch_result3 = ", torch_result3)


    print("torch_result3[0].shape = ", torch_result3[0].shape)



    tensor = tf.reshape( tf.convert_to_tensor(np.array(
    [True, False, True, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False])), (2, 2, 10) )

    tf_result3 = torch_where(tensor)
    # [0]

    print("tf_result3 = ", tf_result3)


    # print("np.allclose(torch_result3.numpy(), tf_result3.numpy()) = ", np.allclose(torch_result3[0].numpy(), tf_result3.numpy()))
    print("np.allclose(torch_result3.numpy(), tf_result3.numpy()) = ", np.allclose(torch_result3[0].numpy(), tf_result3[0].numpy()))
    assert np.allclose(torch_result3[0].numpy(), tf_result3[0].numpy())





