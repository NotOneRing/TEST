import numpy as np

from util.torch_to_tf import torch_sum

import tensorflow as tf
import torch

def test_sum():
    arr = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])


    probs = tf.constant(arr, dtype=tf.float32)

    self_probs_tf = probs /torch_sum(probs, dim=-1, keepdim=True)

    print("self_probs_tf = ", self_probs_tf)



    probs = torch.tensor(arr)

    self_probs = probs / probs.sum(-1, keepdim=True)

    print("self_probs = ", self_probs)

    assert np.allclose(self_probs.numpy(), self_probs_tf.numpy())

    # 示例概率张量


test_sum()






