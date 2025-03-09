import numpy as np
import torch
import tensorflow as tf


def test_MixtureSameFamily():
    np.random.seed(42)
    batch_size = 4
    num_modes = 2
    dim = 3

    means = np.random.randn(batch_size, num_modes, dim)
    scales = np.abs(np.random.randn(batch_size, num_modes, dim)) * 0.5
    logits = np.random.randn(batch_size, num_modes)

    means_torch = torch.tensor(means, dtype=torch.float32)
    scales_torch = torch.tensor(scales, dtype=torch.float32)
    logits_torch = torch.tensor(logits, dtype=torch.float32)

    means = means_torch
    scales = scales_torch
    logits = logits_torch

    import torch.distributions as D

    # Mixture components for PyTorch
    component_distribution = D.Normal(loc=means, scale=scales)
    component_distribution = D.Independent(component_distribution, 1)

    # Unnormalized logits to categorical distribution for mixing the modes
    mixture_distribution = D.Categorical(logits=logits)
    dist_torch = D.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        component_distribution=component_distribution,
    )

    # Option 1: calculate average for dimension modes
    sample_torch = means.mean(dim=1)  # calculate the average for each sample in all modes
    log_prob_torch = dist_torch.log_prob(sample_torch)

    print("Torch Log Probability: ", log_prob_torch)

    means_tf = tf.convert_to_tensor(means, dtype=tf.float32)
    scales_tf = tf.convert_to_tensor(scales, dtype=tf.float32)
    logits_tf = tf.convert_to_tensor(logits, dtype=tf.float32)

    means = means_tf
    scales = scales_tf
    logits = logits_tf

    from util.torch_to_tf import Normal, Independent, Categorical, MixtureSameFamily

    # Mixture components for TensorFlow
    component_distribution_tf = Normal(means, scales)
    component_distribution_tf = Independent(component_distribution_tf, 1)

    # Unnormalized logits to categorical distribution for mixing the modes
    mixture_distribution_tf = Categorical(logits=logits)
    dist_tf = MixtureSameFamily(
        mixture_distribution=mixture_distribution_tf,
        component_distribution=component_distribution_tf,
    )

    # Option 1: calculate average for dimension modes
    sample_tf = tf.reduce_mean(means, axis=1)  # calculate the average for each sample in all modes
    log_prob_tf = dist_tf.log_prob(sample_tf)

    print("TensorFlow Log Probability: ", log_prob_tf)

    assert np.allclose(log_prob_torch, log_prob_tf)










