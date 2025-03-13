import tensorflow as tf
import numpy as np
import torch
import unittest

from util.torch_to_tf import torch_logsumexp


class TestLogSumExp(unittest.TestCase):
    def test_logsumexp(self):
        # Suppose log_prob_x and log_mix_prob are two Tensors
        log_prob_x = tf.constant([[0.1, 0.2], [0.3, 0.4]])  # example
        log_mix_prob = tf.constant([[0.5, -0.5], [0.6, -0.6]])  # example

        log_prob_sum = log_prob_x + log_mix_prob  # [2, 2]

        logsumexp_result_tf = torch_logsumexp(log_prob_sum, dim=-1)

        logsumexp_result_tf2 = tf.math.log( tf.reduce_sum( tf.math.exp(log_prob_sum), axis=-1) )

        log_prob_x_torch = torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # example
        log_mix_prob_torch = torch.tensor([[0.5, -0.5], [0.6, -0.6]])  # example

        log_prob_sum_torch = log_prob_x_torch + log_mix_prob_torch  # [2, 2]

        log_prob_sum_torch2 = torch.log( torch.sum( torch.exp(log_prob_sum_torch), axis=-1) )

        logsumexp_result_torch = torch.logsumexp(log_prob_sum_torch, dim=-1)

        # Compare the results
        self.assertTrue(np.allclose(logsumexp_result_tf.numpy(), logsumexp_result_torch.numpy()))

        self.assertTrue(np.allclose(logsumexp_result_tf.numpy(), log_prob_sum_torch2.numpy()))

        self.assertTrue(np.allclose(logsumexp_result_tf2.numpy(), log_prob_sum_torch2.numpy()))


if __name__ == '__main__':
    unittest.main()
