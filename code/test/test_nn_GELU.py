import torch
import torch.nn as nn

import numpy as np

def test_GELU():
    # 创建一个GELU激活层
    gelu = nn.GELU()

    # 输入张量
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])

    # 应用ELU激活
    output = gelu(input_tensor)
    print(output)



    import tensorflow as tf
    from util.torch_to_tf import nn_GELU

    # 创建一个ELU激活层
    gelu = nn_GELU()

    # 输入张量
    input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

    # 应用ELU激活
    output_tf = gelu(input_tensor_tf)
    print(output_tf)

    assert np.allclose(output, output_tf)



test_GELU()













