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
    x = tf.linspace(0.0, float(steps), steps)  # 生成均匀分布的点
    alphas_cumprod = tf.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2  # 计算累积alpha
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  # 计算beta值
    betas_clipped = tf.clip_by_value(betas, clip_value_min=0.0, clip_value_max=0.999)  # 截断

    return tf.convert_to_tensor(betas_clipped, dtype=dtype)


# def extract(a, t, x_shape):
#     print("sampling.py: extract()")
#     print("22222")

#     b = tf.shape(t)[0]
#     out = tf.gather(a, t, axis=-1)
#     return tf.reshape(out, [b] + [1] * (len(x_shape) - 1))


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


    out = tf.gather(a, t, axis=-1)

    # print("out = ", out)
    # print("out.shape = ", out.shape)
    

    reshape_shape = [b] + [1] * (len(x_shape) - 1)
    # print("reshape_shape = ", reshape_shape)

    # Reshape the output to match x_shape
    return tf.reshape(out, reshape_shape)



def make_timesteps(batch_size, i):
    print("sampling.py: make_timesteps()")
    t = tf.constant([i] * batch_size, dtype=tf.int64)
    return t












