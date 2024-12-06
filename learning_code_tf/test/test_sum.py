import numpy as np


arr = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])

from util.torch_to_tf import torch_sum

import tensorflow as tf

probs = tf.constant(arr, dtype=tf.float32)

self_probs = probs /torch_sum(probs, dim=-1, keepdim=True)

print("self_probs = ", self_probs)


import torch

probs = torch.tensor(arr)

self_probs = probs / probs.sum(-1, keepdim=True)

print("self_probs = ", self_probs)


# 示例概率张量

