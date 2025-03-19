"""
From Diffuser https://github.com/jannerm/diffuser

"""

import tensorflow as tf
import numpy as np

from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, OUTPUT_POSITIONS, OUTPUT_FUNCTION_HEADER


def cosine_beta_schedule(timesteps, s=0.008, dtype=tf.float32):
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """

    if OUTPUT_VARIABLES:
        print("timesteps = ", timesteps)

    if OUTPUT_FUNCTION_HEADER:
        print("cosine_beta_schedule() called")

    steps = timesteps + 1
    x = np.linspace(0.0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0.0, a_max=0.999)
    return tf.convert_to_tensor(betas_clipped, dtype=dtype)



def extract(a, t, x_shape):
    
    if OUTPUT_FUNCTION_HEADER:
        print("sampling.py: extract()")

    b = t.shape[0]
    
    b = int(b)


    from util.torch_to_tf import torch_gather
    out2 = torch_gather(a, -1, t)


    reshape_shape = [b] + [1] * (len(x_shape) - 1)


    # Reshape the output to match x_shape
    return tf.reshape(out2, reshape_shape)



def make_timesteps(batch_size, i):
    if OUTPUT_FUNCTION_HEADER:
        print("sampling.py: make_timesteps()")
    
    from util.torch_to_tf import torch_full
    t2 = torch_full((batch_size,), i, dtype=tf.int64)

    return t2












