
from util.torch_to_tf import torch_repeat_interleave

import torch

import numpy as np

# tensor_np = np.array([1, 2, 3, 4]).reshape(2, 2)

tensor_np = np.array(range(27)).reshape(3, 3, 3)

# repeats_np = np.array([2, 3, 4, 1])

repeats = 3

# 测试
tensor = torch.tensor(tensor_np)
# repeats = torch.tensor(repeats_np)

# output = torch.repeat_interleave(tensor, repeats)
torch_output = torch.repeat_interleave(tensor, repeats, dim=0)

print(torch_output)

print("output.shape = ", torch_output.shape)

import tensorflow as tf

tensor = tf.convert_to_tensor(tensor_np)
# repeats = tf.convert_to_tensor(repeats_np)

# output = torch_repeat_interleave(tensor, repeats)
tf_output = torch_repeat_interleave(tensor, repeats, dim=0)

print(tf_output)

print("output.shape = ", tf_output.shape)


from util.func_pytest import compare_result


# print(compare_result(torch_output, tf_output))
print(compare_result(tf_output, torch_output))


