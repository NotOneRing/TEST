


import torch

a = torch.tensor([[1, 2], [3, 4]]).reshape(1, 1, 4)
b = torch.tensor([[5, 6], [7, 8]]).reshape(1, 1, 4)

result = torch.hstack((a, b))
print(result)

print(result.shape)

# 输出：
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])

import tensorflow as tf

from util.torch_to_tf import torch_hstack, torch_tensor_view

a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

a = torch_tensor_view(a, 1, 1, 4)
b = torch_tensor_view(a, 1, 1, 4)

result = torch_hstack((a, b))
print(result)
# 输出：
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])



