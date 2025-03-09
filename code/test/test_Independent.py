


import torch
from torch.distributions import Normal, Independent

# # define two independent Normal distributions
# normal1 = Normal(torch.zeros(5), torch.ones(5))  # 0 as mean, and 1 as std
# normal2 = Normal(torch.ones(5), torch.ones(5))  # 0 as mean, and 1 as std

# # wrap them into independent distributions
# independent_distribution = Independent(normal1, reinterpreted_batch_ndims=1)

# # draw samples
# samples = independent_distribution.sample()
# print(f"Independent Samples: {samples}")


# import torch
# from torch.distributions import Independent
# , Normal 

import tensorflow as tf

import numpy as np

from util.torch_to_tf import Normal, Independent, torch_zeros, torch_ones


# define a batch of data, each data point is two independent normal distributions
base_distribution = Normal(torch_zeros(3, 2), torch_ones(3, 2))  # 3 data points, each point has 2 independent normal distributions

# Use Independent, regard them as independent distributions
independent_distribution = Independent(base_distribution, reinterpreted_batch_ndims=1)

# draw samples
samples = independent_distribution.sample()
print(f"1Independent Samples (Batch): {samples}")


# x = tf.convert_to_tensor( np.array([1.0, 1.0, 2.0]), dtype=np.float32 )

x = samples

# calculate
log_prob = independent_distribution.log_prob(x)  # log probability to x
print(f"1Log probability at x = {x}: {log_prob.numpy()}")















import torch

from torch.distributions import Normal, Independent


# Define a batch of data points. Each data point is two independent normal distributions.
base_distribution = Normal(torch.zeros(3, 2), torch.ones(3, 2))  # 3 data points, each point has 2 independent normal distributions

# Use Independent, regard them as independent distributions
independent_distribution = Independent(base_distribution, reinterpreted_batch_ndims=1)

# # draw samples
samples = torch.tensor(samples.numpy())
print(f"2Independent Samples (Batch): {samples}")

# samples = independent_distribution.sample()
# print(f"2Independent Samples (Batch): {samples}")

# x = torch.tensor(x.numpy())
x = torch.tensor(samples)


log_prob = independent_distribution.log_prob(x)  


# log_prob = log_normal_pdf(x, mu, sigma)
print(f"2Log probability at x = {x}: {log_prob.numpy()}")













# import torch
# from torch.distributions import Normal, Independent

# # define one three-dimensional normal distribution
# base_distribution = Normal(torch.zeros(2, 3, 4), torch.ones(2, 3, 4))  # shape of (2, 3, 4)

# # use Independent to handle the independent distribution within each batch
# independent_distribution = Independent(base_distribution, reinterpreted_batch_ndims=2)

# # draw samples
# samples = independent_distribution.sample()
# print(f"Independent Samples (3D Batch): {samples}")































