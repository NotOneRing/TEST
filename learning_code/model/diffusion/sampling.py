"""
From Diffuser https://github.com/jannerm/diffuser

"""

import torch
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """

    print("sampling.py: cosine_beta_schedule()", flush = True)
    # print("sampling.py: cosine_beta_schedule()", flush = True)

    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def extract(a, t, x_shape):

    print("sampling.py: extract()", flush = True)
    # print("11111", flush=True)
    # print("1a = ", a, flush=True)
    # print("1type(a) = ", type(a), flush=True)

    b, *_ = t.shape
    print("a = ", a, flush=True)
    print("type(a) = ", type(a), flush=True)
    print("a.shape = ", a.shape, flush=True)

    print("t = ", t, flush=True)
    print("type(t) = ", type(t), flush=True)
    print("t.shape = ", t.shape, flush=True)

    print("x_shape = ", x_shape, flush=True)
    print("type(x_shape) = ", type(x_shape), flush=True)
    # print("x_shape.shape = ", x_shape.shape, flush=True)


    out = a.gather(-1, t)
    print("out = ", out, flush=True)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))





def make_timesteps(batch_size, i, device):

    print("sampling.py: make_timesteps()", flush = True)

    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t





