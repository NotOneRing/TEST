import torch

import numpy as np

import tensorflow as tf

arr = np.array([1.0])


from util.func_pytest import np_to_tf, np_to_torch


tf_arr = np_to_tf(arr)

torch_arr = np_to_torch(arr)


print("torch_arr.item() = ", torch_arr.item())
print("type(torch_arr.item()) = ", type(torch_arr.item()))

from util.torch_to_tf import torch_tensor_item

print("torch_tensor_item(tf_arr)", torch_tensor_item(tf_arr))

print("type( torch_tensor_item(tf_arr) )", type( torch_tensor_item(tf_arr) ) )


