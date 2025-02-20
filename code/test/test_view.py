import torch

import numpy as np

import tensorflow as tf

from util.torch_to_tf import torch_tensor_view


def test_view():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    result1 = tensor.view(3, 3)

    print(result1)
    print(result1.shape)

    result2 = tensor.view(3, 3, 1)


    print("result2 = ", result2)
    print("result2.shape = ", result2.shape)

    print("tensor = ", tensor)
    print("tensor.shape = ", tensor.shape)



    result3 = tensor.view([3, 3, 1])


    print("result3 = ", result3)
    print("result3.shape = ", result3.shape)

    print("tensor = ", tensor)
    print("tensor.shape = ", tensor.shape)


    tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    tensor = tf.convert_to_tensor(tensor)


    result_tf1 = torch_tensor_view(tensor, 3, 3)
    print("result_tf1 = ", result_tf1)


    result_tf2 = torch_tensor_view(tensor, [3, 3, 1])
    print("result_tf2 = ", result_tf2)


    result_tf3 = torch_tensor_view(tensor, 3, 3, 1)
    print("result_tf3 = ", result_tf3)

    assert np.allclose(result1.numpy(), result_tf1.numpy())
    assert np.allclose(result2.numpy(), result_tf2.numpy())
    assert np.allclose(result3.numpy(), result_tf3.numpy())


test_view()


