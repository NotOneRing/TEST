import torch
import torch.nn.functional as F


import tensorflow as tf

from util.torch_to_tf import torch_mse_loss

import numpy as np

def test_mse_loss():

    # 示例数据
    input = torch.tensor([0.5, 1.0, 1.5])
    target = torch.tensor([1.0, 1.0, 1.0])

    # 计算 MSE 损失
    loss = F.mse_loss(input, target, reduction='mean')
    print(loss)  # 输出均方误差损失

    print( type(loss) )  

    # 示例数据
    input = tf.constant([0.5, 1.0, 1.5])
    target = tf.constant([1.0, 1.0, 1.0])

    # 计算 MSE 损失
    loss_tf = torch_mse_loss(input, target, reduction='mean')
    print(loss_tf)  # 输出均方误差损失

    print( type(loss_tf) )

    assert np.allclose(loss_tf.numpy(), loss.numpy())




test_mse_loss()





