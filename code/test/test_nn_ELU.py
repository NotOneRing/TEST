import torch
import torch.nn as nn

import numpy as np

def test_ELU():
    # 创建一个ELU激活层
    elu = nn.ELU()

    # 输入张量
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])

    # 应用ELU激活
    output = elu(input_tensor)
    print(output)



    import tensorflow as tf
    from util.torch_to_tf import nn_ELU

    # 创建一个ELU激活层
    elu = nn_ELU()

    # 输入张量
    input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

    # 应用ELU激活
    output_tf = elu(input_tensor_tf)
    print(output_tf)

    assert np.allclose(output, output_tf)



test_ELU()


