import torch

import tensorflow as tf

import numpy as np

input = np.array(range(1, 10), dtype = np.float32).reshape(3, 3)


torch_input = torch.tensor(input)

a = torch.softmax(torch_input, dim=1)

tf_input = tf.convert_to_tensor(input)

# b = tf.nn.softmax(tf_input, axis=1)

from util.torch_to_tf import torch_softmax

b = torch_softmax(torch_input, dim=1)

print("a = ", a)

print("b = ", b)









