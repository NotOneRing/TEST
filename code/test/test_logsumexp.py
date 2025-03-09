import tensorflow as tf

from util.torch_to_tf import torch_logsumexp

import numpy as np

import torch


def test_logsumexp():
    # Suppose log_prob_x and log_mix_prob are two Tensors
    log_prob_x = tf.constant([[0.1, 0.2], [0.3, 0.4]])  # example
    log_mix_prob = tf.constant([[0.5, -0.5], [0.6, -0.6]])  # example

    # merge two tensors
    log_prob_sum = log_prob_x + log_mix_prob  # [2, 2]

    # calculate logsumexp, get summation in the last dimension
    logsumexp_result_tf = torch_logsumexp(log_prob_sum, dim=-1)

    # output the result
    print(logsumexp_result_tf)






    # Suppose log_prob_x and log_mix_prob are two Tensors
    log_prob_x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # example
    log_mix_prob = torch.tensor([[0.5, -0.5], [0.6, -0.6]])  # example

    # merge two tensors
    log_prob_sum = log_prob_x + log_mix_prob  # [2, 2]

    # calculate logsumexp, get summation in the last dimension
    logsumexp_result = torch.logsumexp(log_prob_sum, dim=-1)

    # output the result
    print(logsumexp_result)

    assert np.allclose(logsumexp_result_tf.numpy(), logsumexp_result.numpy())





test_logsumexp()

