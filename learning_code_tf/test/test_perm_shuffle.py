import torch

import tensorflow as tf

sz = 10

perm = torch.randperm(sz)
print("torch.randperm = ", perm)

perm = tf.random.shuffle(tf.range(sz))
print("tf.random.shuffle = ", perm)


