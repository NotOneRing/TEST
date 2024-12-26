import tensorflow as tf

import numpy as np

from util.torch_to_tf import torch_nanmean

def test_nanmean():
    arr = np.array([1.1, 1.2, 1.3, float('nan')])

    log_arr = np.array([0.1, 0.2, 0.3, 0.4])

    ratio = tf.constant(arr, dtype=tf.float64)
    logratio = tf.constant(log_arr, dtype=tf.float64)


    kl_difference = (ratio - 1) - logratio

    print("kl_difference = ", kl_difference)

    # kl_difference_no_nan = tf.boolean_mask(kl_difference, ~tf.math.is_nan(kl_difference))

    # print("kl_difference_no_nan = ", kl_difference_no_nan)

    # approx_kl = tf.reduce_mean(kl_difference_no_nan)

    tf_approx_kl = torch_nanmean(kl_difference)

    print("Approximate KL Divergence:", tf_approx_kl.numpy())


    import torch

    ratio = torch.tensor(arr)

    logratio = torch.tensor(log_arr)

    kl_difference = (ratio - 1) - logratio

    print("kl_difference = ", kl_difference)

    approx_kl = kl_difference.nanmean()

    print("Approximate KL Divergence:", approx_kl.numpy())


    assert np.allclose(tf_approx_kl.numpy(), approx_kl.numpy())


test_nanmean()





