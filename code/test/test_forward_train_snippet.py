import numpy as np
import torch
import tensorflow as tf

# Step 1: use numpy to construct data
np.random.seed(42)
batch_size = 4
num_modes = 2
dim = 3

means = np.random.randn(batch_size, num_modes, dim)
scales = np.abs(np.random.randn(batch_size, num_modes, dim)) * 0.5
logits = np.random.randn(batch_size, num_modes)

# Step 2: convert numpy into torch and tensorflow
means_torch = torch.tensor(means, dtype=torch.float32)
scales_torch = torch.tensor(scales, dtype=torch.float32)
logits_torch = torch.tensor(logits, dtype=torch.float32)

means = means_torch
scales = scales_torch
logits = logits_torch

import torch.distributions as D

# if deterministic:
#     # low-noise for all Gaussian dists
#     scales = torch.ones_like(means) * 1e-4

# mixture components - make sure that `batch_shape` for the distribution is equal to (batch_size, num_modes) since MixtureSameFamily expects this shape
# Each mode has mean vector of dim T*D

# print("pos1")
component_distribution = D.Normal(loc=means, scale=scales)

# print("component_distribution = ", component_distribution)

component_distribution = D.Independent(component_distribution, 1)

# print("component_distribution = ", component_distribution)

# print("pos2")

component_entropy = component_distribution.entropy()
approx_entropy = torch.mean(
    torch.sum(logits.softmax(-1) * component_entropy, dim=-1)
)
std = torch.mean(torch.sum(logits.softmax(-1) * scales.mean(-1), dim=-1))

# print("pos3")

# unnormalized logits to categorical distribution for mixing the modes
mixture_distribution = D.Categorical(logits=logits)
dist = D.MixtureSameFamily(
    mixture_distribution=mixture_distribution,
    component_distribution=component_distribution,
)

print("std = ", std)

print("approx_entropy = ", approx_entropy)

print("dist.log_prob() = ", dist.log_prob())





from util.torch_to_tf import Normal, Independent, Categorical, MixtureSameFamily



means_tf = tf.convert_to_tensor(means, dtype=tf.float32)
scales_tf = tf.convert_to_tensor(scales, dtype=tf.float32)
logits_tf = tf.convert_to_tensor(logits, dtype=tf.float32)

means = means_tf
scales = scales_tf
logits = logits_tf

# if deterministic:
#     # low-noise for all Gaussian dists
#     scales = tf.ones_like(means) * 1e-4

# mixture components - make sure that `batch_shape` for the distribution is equal to (batch_size, num_modes) since MixtureSameFamily expects this shape
# Each mode has mean vector of dim T*D

# component_distribution = tfp.distributions.Normal(loc=means, scale=scales)
component_distribution = Normal(means, scales)

component_distribution = Independent(component_distribution, 1)

component_entropy = component_distribution.entropy()



approx_entropy = tf.reduce_mean(
    tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * component_entropy, axis=-1)
)

std = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * tf.reduce_mean(scales, axis=-1), axis=-1))

# Unnormalized logits to categorical distribution for mixing the modes
mixture_distribution = Categorical(logits=logits)


print("std = ", std)

print("approx_entropy = ", approx_entropy)

dist = MixtureSameFamily(
    mixture_distribution=mixture_distribution,
    component_distribution=component_distribution,
)

print("dist.log_prob() = ", dist.log_prob())




# log_prob = dist.log_prob( tf.reshape(actions, [B, -1]) )


