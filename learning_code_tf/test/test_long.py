
from util.torch_to_tf import torch_tensor_long

from util.func_pytest import np_to_tf, np_to_torch, gen_3d_int

import tensorflow as tf

import torch

def test_long():
    test_case = gen_3d_int()[0]
    tf_tensor = np_to_tf(test_case)
    torch_tensor = np_to_torch(test_case)
    tf_tensor = torch_tensor_long(tf_tensor)
    torch_tensor = torch_tensor.long()
    assert torch_tensor.dtype == torch.long
    assert tf_tensor.dtype == tf.int64


test_long()
