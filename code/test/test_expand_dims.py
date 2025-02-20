import tensorflow as tf

import numpy as np

import torch

a = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])

print("a[:, None] = ", a[:, None])

print("a[:, None].shape = ", a[:, None].shape)

b = torch.tensor(a)

b = b[:, None]

print("b = ", b)

print("b.shape = ", b.shape)


from util.torch_to_tf import torch_tensor

c = torch_tensor(a)

c = c[:, None]

print("c.shape = ", c.shape)
