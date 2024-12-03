
import tensorflow as tf

import copy


def tf_index_gather(input_tensor, dim, index_tensor):
    """
    Mimics the behavior of indexing in PyTorch. 
    Specifically:
        - out[i][j][k] = input[index[i][j][k]][j][k]  if dim == 0
        - out[i][j][k] = input[i][index[i][j][k]][k]  if dim == 1
        - out[i][j][k] = input[i][j][index[i][j][k]]  if dim == 2
    
    Args:
        input_tensor (tf.Tensor): The input tensor from which to gather values.
        index_tensor (tf.Tensor): The indices tensor.
        dim (int): The dimension along which to gather the values.
    
    Returns:
        tf.Tensor: The output tensor with the gathered values.
    """

    assert input_tensor.shape.as_list() == index_tensor.shape.as_list(), "input_tensor.shape is not equal to index_tensor.shape"

    index_array = index_tensor.numpy()

    input_array = input_tensor.numpy()

    dim_list = input_tensor.shape.as_list()

    dim_number = len(input_tensor.shape)

    cur_index = [0] * dim_number
    
    import numpy as np
    output_matrix = np.zeros(dim_list, dtype=np.int64)

    import math
    total_number = math.prod(dim_list)

    count_number = 0

    while count_number < total_number:

        for cur_dim_index in range(dim_list[dim_number - 1]):
            cur_index[dim_number-1] = cur_dim_index
            dim_true_index = index_array[ tuple(cur_index) ]
            dim_ori_index = cur_index[dim]
            cur_index[dim] = dim_true_index
            val = input_array[ tuple(cur_index) ]
            cur_index[dim] = dim_ori_index
            output_matrix[tuple(cur_index)] = val
            count_number += 1

        cur_index[dim_number - 1] += 1

        for i in range(dim_number - 1, 0, -1):
            if cur_index[i] > dim_list[i] - 1:
                cur_index[i] -= (dim_list[i])
                cur_index[i-1] += 1
            else:
                break

        if cur_index[0] > dim_list[0] - 1:
            break

    output_matrix = tf.convert_to_tensor(output_matrix)
    
    return output_matrix



# Example input tensor (shape: [3, 3, 3])
input_tensor = tf.constant([[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]],
                            
                            [[10, 11, 12],
                             [13, 14, 15],
                             [16, 17, 18]],
                            
                            [[19, 20, 21],
                             [22, 23, 24],
                             [25, 26, 27]]], dtype=tf.int32)

# Example index tensor (shape: [3, 3, 3]) -- Indices to gather from the input tensor
index_tensor = tf.constant([[[0, 2, 1],
                             [1, 0, 2],
                             [2, 1, 0]],
                            
                            [[2, 0, 1],
                             [0, 2, 1],
                             [1, 2, 0]],
                            
                            [[1, 2, 0],
                             [2, 1, 0],
                             [0, 1, 2]]], dtype=tf.int32)

# Gather along the 0th dimension (dim == 0)
out_dim_0 = tf_index_gather(input_tensor, 0, index_tensor)
print("Gather along dim 0:")
print(out_dim_0.numpy())

# Gather along the 1st dimension (dim == 1)
out_dim_1 = tf_index_gather(input_tensor, 1, index_tensor)
print("\nGather along dim 1:")
print(out_dim_1.numpy())

# Gather along the 2nd dimension (dim == 2)
out_dim_2 = tf_index_gather(input_tensor, 2, index_tensor)
print("\nGather along dim 2:")
print(out_dim_2.numpy())


import torch

input_tensor = torch.tensor(input_tensor.numpy(), dtype = torch.int64)
index_tensor = torch.tensor(index_tensor.numpy(), dtype = torch.int64)

# Gather along the 0th dimension (dim == 0)
out_dim_0 = torch.gather(input_tensor, 0, index_tensor)
print("torch Gather along dim 0:")
print(out_dim_0.numpy())

# Gather along the 1st dimension (dim == 1)
out_dim_1 = torch.gather(input_tensor, 1, index_tensor)
print("\ntorch Gather along dim 1:")
print(out_dim_1.numpy())

# Gather along the 2nd dimension (dim == 2)
out_dim_2 = torch.gather(input_tensor, 2, index_tensor)
print("\ntorch Gather along dim 2:")
print(out_dim_2.numpy())









































