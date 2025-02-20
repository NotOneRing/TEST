import torch
import torch.nn as nn
import numpy as np



def test_Softplus():
    # 创建一个ELU激活层
    softplus = nn.Softplus()

    # 输入张量
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])

    # 应用ELU激活
    output = softplus(input_tensor)
    print(output)



    import tensorflow as tf
    from util.torch_to_tf import nn_Softplus

    # 创建一个ELU激活层
    softplus = nn_Softplus()

    # 输入张量
    input_tensor_tf = tf.constant([-1.0, 0.0, 1.0])

    # 应用ELU激活
    output_tf = softplus(input_tensor_tf)
    print(output_tf)

    assert np.allclose(output, output_tf)



test_Softplus()







