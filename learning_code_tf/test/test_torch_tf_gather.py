
import tensorflow as tf

import copy

from util.torch_to_tf import torch_gather

import torch

from util.func_pytest import compare_result, np_to_tf, np_to_torch, gen_3d_int, gen_3d_index



def test_gather():

    input_tensor = gen_3d_int()[0]
    index_tensor = gen_3d_index()[0]



    tf_input_tensor = np_to_tf(input_tensor)

    tf_index_tensor = np_to_tf(index_tensor)


    torch_input_tensor = np_to_torch(input_tensor)

    torch_index_tensor = np_to_torch(index_tensor)



    # Gather along the 0th dimension (dim == 0)
    out_tf_0 = torch_gather(tf_input_tensor, 0, tf_index_tensor)

    out_torch_0 = torch.gather(torch_input_tensor, 0, torch_index_tensor)

    print("compare = ", compare_result(out_torch_0, out_tf_0))
    assert compare_result(out_torch_0, out_tf_0)





    # Gather along the 1st dimension (dim == 1)
    out_tf_1 = torch.gather(torch_input_tensor, 1, torch_index_tensor)

    out_torch_1 = torch_gather(tf_input_tensor, 1, tf_index_tensor)


    print("compare = ", compare_result(out_torch_1, out_tf_1))
    assert compare_result(out_torch_1, out_tf_1)




    # Gather along the 2nd dimension (dim == 2)
    out_torch_2 = torch.gather(torch_input_tensor, 2, torch_index_tensor)
    out_tf_2 = torch_gather(tf_input_tensor, 2, tf_index_tensor)


    print("compare = ", compare_result(out_torch_2, out_tf_2))
    assert compare_result(out_torch_2, out_tf_2)








































