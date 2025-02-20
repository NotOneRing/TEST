

import torch

from util.func_pytest import *


import tensorflow as tf

from util.torch_to_tf import torch_triu

def test_triu():

    matrix = gen_2d_int()[0]

    # 创建一个 3x3 矩阵
    torch_matrix = np_to_torch(matrix)

    tf_matrix = np_to_tf(matrix)


    # 创建一个 3x3 矩阵
    tf_matrix = np_to_tf(matrix)



    # 提取主对角线及其以上的部分
    triu_matrix = torch.triu(torch_matrix, diagonal=0)
    print(triu_matrix)
    # 输出:
    # tensor([[1, 2, 3],
    #         [0, 5, 6],
    #         [0, 0, 9]])




    # 创建一个上三角矩阵
    upper_triangle_matrix = torch_triu(tf_matrix, diagonal=0)
    print(upper_triangle_matrix)


    assert np.allclose(triu_matrix.numpy(), upper_triangle_matrix.numpy())

    print("1")



    # 提取主对角线以上的部分
    triu_matrix = torch.triu(torch_matrix, diagonal=1)
    print(triu_matrix)
    # 输出:
    # tensor([[0, 2, 3],
    #         [0, 0, 6],
    #         [0, 0, 0]])


    upper_triangle_matrix = torch_triu(tf_matrix, diagonal=1)
    print(upper_triangle_matrix)


    assert np.allclose(triu_matrix.numpy(), upper_triangle_matrix.numpy())

    print("2")




    # 提取主对角线以下的部分
    triu_matrix = torch.triu(torch_matrix, diagonal=-1)
    print(triu_matrix)
    # 输出:
    # tensor([[1, 2, 3],
    #         [4, 5, 6],
    #         [0, 8, 9]])

    upper_triangle_matrix = torch_triu(tf_matrix, diagonal=-1)
    print(upper_triangle_matrix)



    assert np.allclose(triu_matrix.numpy(), upper_triangle_matrix.numpy())


    print("3")


test_triu()












