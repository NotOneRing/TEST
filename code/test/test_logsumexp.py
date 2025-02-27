import tensorflow as tf

from util.torch_to_tf import torch_logsumexp

import numpy as np

import torch


def test_logsumexp():
    # 假设 log_prob_x 和 log_mix_prob 是两个 Tensor
    log_prob_x = tf.constant([[0.1, 0.2], [0.3, 0.4]])  # 示例数据
    log_mix_prob = tf.constant([[0.5, -0.5], [0.6, -0.6]])  # 示例数据

    # 合并两个张量
    log_prob_sum = log_prob_x + log_mix_prob  # [2, 2]

    # 计算 logsumexp，在最后一个维度进行求和
    logsumexp_result_tf = torch_logsumexp(log_prob_sum, dim=-1)

    # 输出结果
    print(logsumexp_result_tf)






    # 假设 log_prob_x 和 log_mix_prob 是两个 Tensor
    log_prob_x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # 示例数据
    log_mix_prob = torch.tensor([[0.5, -0.5], [0.6, -0.6]])  # 示例数据

    # 合并两个张量
    log_prob_sum = log_prob_x + log_mix_prob  # [2, 2]

    # 计算 logsumexp，在最后一个维度进行求和
    logsumexp_result = torch.logsumexp(log_prob_sum, dim=-1)

    # 输出结果
    print(logsumexp_result)

    assert np.allclose(logsumexp_result_tf.numpy(), logsumexp_result.numpy())





test_logsumexp()

