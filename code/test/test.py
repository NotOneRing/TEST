# import torch

# seed = 42
# torch.manual_seed(seed)

# shape = (2, 3)
# min_val, max_val = -1.0, 1.0

# # use torch.empty and uniform_
# torch_tensor = torch.empty(shape).uniform_(min_val, max_val)
# print("PyTorch Uniform Tensor (uniform_):\n", torch_tensor)

# # use torch.rand and manually scale
# rand_tensor = (min_val - max_val) * torch.rand(shape) + max_val
# print("PyTorch rand Tensor (scaled):\n", rand_tensor)

# # check the difference
# print("Are the tensors equal?", torch.allclose(torch_tensor, rand_tensor))

import tensorflow as tf


a = tf.Tensor([1,2,3])
b = tf.Tensor([4,5,6])

print("a * b = ", a * b)










