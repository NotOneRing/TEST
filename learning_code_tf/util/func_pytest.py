import numpy as np

import tensorflow as tf

import torch






def gen_1d_index():
    test_cases = []

    return test_cases




def gen_2d_index():
    test_cases = []

    return test_cases





def gen_3d_index():
    test_cases = []

    case1 = np.array([[[0, 2, 1],
                       [1, 0, 2],
                       [2, 1, 0]],
                      [[2, 0, 1],
                       [0, 2, 1],
                       [1, 2, 0]],
                    
                      [[1, 2, 0],
                       [2, 1, 0],
                       [0, 1, 2]]])
    test_cases.append(case1)

    return test_cases









def gen_1d_int():
    test_cases = []    
    case1 = np.array(range(27)).reshape(3, 3, 3)
    test_cases.append(case1)

    return test_cases





def gen_2d_int():
    test_cases = []
    
    case1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    test_cases.append(case1)

    return test_cases





def gen_3d_int():
    test_cases = []

    case1 = np.array([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]],
                      [[10, 11, 12],
                       [13, 14, 15],
                       [16, 17, 18]],
                      [[19, 20, 21],
                       [22, 23, 24],
                       [25, 26, 27]]])
    test_cases.append(case1)

    return test_cases





def gen_1d_float():
    test_cases = []

    return test_cases





def gen_2d_float():
    test_cases = []

    return test_cases





def gen_3d_float():
    test_cases = []

    return test_cases




def np_to_tf(input):
    return tf.convert_to_tensor(input)





def np_to_torch(input):
    return torch.tensor(input)






def compare_result(tf_tensor, torch_tensor):
    if isinstance(tf_tensor, torch.Tensor):
        temp = tf_tensor
        tf_tensor = torch_tensor
        torch_tensor = temp
    
    return  np.allclose(tf_tensor.numpy(), torch_tensor.numpy())





























