import tensorflow as tf

import torch

h = 5

pad = 10

eps = 1.0 / (h + pad)

arange = torch.linspace(
    -1.0 + eps, 1.0 - eps, h + 2 * pad
)[:h]


print("1:torch: arange = ", arange)

arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)

print("2:torch: arange = ", arange)

print("2:torch: arange.shape = ", arange.shape)

arange = tf.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad)[:h]

print("1:tf: arange = ", arange)

arange = tf.reshape(arange, (1, h, 1))

print("2:tf: arange.shape = ", arange.shape)

arange = tf.tile(arange, [h, 1, 1])

print("2:tf: arange = ", arange)


out_scale = torch.tensor([[0.1, 0.2, 0.3],
             [0.4, 0.5, 0.6]])


print("out_scale.shape = ", out_scale.shape)


B = 3
horizon_steps = 2

out_scale = out_scale.repeat(B, horizon_steps)

print("out_scale = ", out_scale)

print("out_scale.shape = ", out_scale.shape)




