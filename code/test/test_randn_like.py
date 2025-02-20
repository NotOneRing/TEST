

from util.torch_to_tf import torch_tensor, torch_randn_like

import numpy as np
import tensorflow as tf
import torch


def test_randn_like():
    np_arr = np.array(range(9)).reshape(3, 3).astype(np.float32)

    # .astype(np.float32)


    tensor_torch = torch.tensor(np_arr)

    tensor_tf = torch_tensor(np_arr)

    tf_tensor_randn_like = torch_randn_like(tensor_tf)

    print("tf_tensor_randn_like = ", tf_tensor_randn_like)

    torch_tensor_randn_like = torch.randn_like(tensor_torch)

    print("torch_tensor_randn_like = ", torch_tensor_randn_like)


test_randn_like()





























