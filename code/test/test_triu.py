

import torch

from util.func_pytest import *


import tensorflow as tf

from util.torch_to_tf import torch_triu

def test_triu():

    matrix = gen_2d_int()[0]

    # create a matrix of 3x3
    torch_matrix = np_to_torch(matrix)

    tf_matrix = np_to_tf(matrix)


    # create a matrix of 3x3
    tf_matrix = np_to_tf(matrix)



    # Extract the main diagonal and the upper triangular part of the matrix.
    triu_matrix = torch.triu(torch_matrix, diagonal=0)
    print(triu_matrix)
    # output:
    # tensor([[1, 2, 3],
    #         [0, 5, 6],
    #         [0, 0, 9]])




    # Create a matrix with elements in the upper triangular part
    upper_triangle_matrix = torch_triu(tf_matrix, diagonal=0)
    print(upper_triangle_matrix)


    assert np.allclose(triu_matrix.numpy(), upper_triangle_matrix.numpy())

    print("1")


    # diagonal is positive, extract from the upper triangular part above the main diagonal of the matrix.
    triu_matrix = torch.triu(torch_matrix, diagonal=1)
    print(triu_matrix)
    # output:
    # tensor([[0, 2, 3],
    #         [0, 0, 6],
    #         [0, 0, 0]])


    upper_triangle_matrix = torch_triu(tf_matrix, diagonal=1)
    print(upper_triangle_matrix)


    assert np.allclose(triu_matrix.numpy(), upper_triangle_matrix.numpy())

    print("2")




    # diagonal is negative, extract from the lower triangular part below the main diagonal of the matrix.
    triu_matrix = torch.triu(torch_matrix, diagonal=-1)
    print(triu_matrix)
    # output:
    # tensor([[1, 2, 3],
    #         [4, 5, 6],
    #         [0, 8, 9]])

    upper_triangle_matrix = torch_triu(tf_matrix, diagonal=-1)
    print(upper_triangle_matrix)



    assert np.allclose(triu_matrix.numpy(), upper_triangle_matrix.numpy())


    print("3")


test_triu()












