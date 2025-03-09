import torch
from torch.distributions import Normal

# define the mean and standard deviation
mean = torch.tensor([0.0, 1.0])
std = torch.tensor([1.0, 0.5])

# create a Normal distribution
dist = Normal(mean, std)

# sample
sample = dist.sample()  # generate one sample
print(f"Sample: {sample}")

# calculate the probability density function(PDF) value
log_prob = dist.log_prob(torch.tensor([1.0, 1.0]))  # probability density function corresponding to [1.0, 1.0]
print(f"Log probability: {log_prob}")





import tensorflow as tf

import numpy as np

from util.torch_to_tf import Normal

# def log_normal_pdf(x, mean, std):
#     """
#     Compute the log probability density function (PDF) of a normal distribution.

#     Args:
#         x: The point at which to evaluate the probability density.
#         mean: The mean of the normal distribution.
#         std: The standard deviation of the normal distribution.

#     Returns:
#         The log probability density.
#     """
#     # Compute the log of the probability density function (PDF)
#     log_pdf = -tf.math.log(std * tf.math.sqrt(2 * tf.constant(np.pi))) - 0.5 * ((x - mean) ** 2) / (std ** 2)
#     return log_pdf



# # test

mu = tf.convert_to_tensor(mean.numpy(), dtype=np.float32)  # mean
sigma = tf.convert_to_tensor(std.numpy(), dtype=np.float32)  # std

# Suppose that we need to calculate the log probability density function at point x = 1
x = tf.convert_to_tensor( np.array([1.0, 1.0]), dtype=np.float32 )

dist = Normal(mu, sigma)

# sample
sample = dist.sample()  # generate one sample
print(f"Sample: {sample}")


log_prob = dist.log_prob(x)  # probability density corresponding to x

# calculate
# log_prob = log_normal_pdf(x, mu, sigma)
print(f"Log probability at x = {x}: {log_prob.numpy()}")


print("log_prob = ", log_prob)

print("type(log_prob) = ", log_prob)



