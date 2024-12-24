"""
From Diffuser https://github.com/jannerm/diffuser

"""

import tensorflow as tf
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008, dtype=tf.float32):
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """

    print("timesteps = ", timesteps)

    print("cosine_beta_schedule() called")

    steps = timesteps + 1
    x = np.linspace(0.0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0.0, a_max=0.999)
    return tf.convert_to_tensor(betas_clipped, dtype=dtype)



def extract(a, t, x_shape):
    print("sampling.py: extract()")

    # b = tf.shape(t)[0]
    # b = t.get_shape().as_list()[0]
    b = t.shape[0]
    
    # print("tf.shape(t):", tf.shape(t))  # 返回形状
    # print("tf.shape(t)[0]:", tf.shape(t)[0])  # 直接获取第一个维度

    # print("int(b.numpy()) = ", int(b.numpy()))
    # print("int(b) = ", int(b))

    # transfer from tensor to numpy
    # b = int(b.numpy())
    b = int(b)

    
    # print("t.shape = ", t.shape)

    # print("b = ", b)
    # print("a = ", a)
    # print("type(a) = ", type(a))
    # print("a.shape = ", a.shape)

    # print("t = ", t)
    # print("type(t) = ", type(t))
    # print("t.shape = ", t.shape)

    # print("x_shape = ", x_shape)
    # print("type(x_shape) = ", type(x_shape))

    # print("a = ", a)
    # print("t = ", t)

    # out = tf.gather(a, t, axis=-1)

    from util.torch_to_tf import torch_gather
    out2 = torch_gather(a, -1, t)

    # print("out = ", out)

    # print("out2 = ", out2)

    # assert np.allclose( out.numpy(), out2.numpy()) , "torch_gather have different result from tf.gather"

    # print("out = ", out)
    # print("out.shape = ", out.shape)
    

    reshape_shape = [b] + [1] * (len(x_shape) - 1)
    # print("reshape_shape = ", reshape_shape)

    # Reshape the output to match x_shape
    return tf.reshape(out2, reshape_shape)



def make_timesteps(batch_size, i):
    print("sampling.py: make_timesteps()")
    # t = tf.constant([i] * batch_size, dtype=tf.int64)

    from util.torch_to_tf import torch_full
    t2 = torch_full((batch_size,), i, dtype=tf.int64)
    # print("t = ", t)
    # print("t2 = ", t2)
    # print( "type(t) = ", type(t) )
    # print( "type(t2) = ", type(t2) )

    # assert np.allclose(t.numpy(), t2.numpy()), "torch_full have different result from tf_make_timesteps"

    return t2












