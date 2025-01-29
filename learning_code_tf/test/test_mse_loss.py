import torch
import torch.nn.functional as F


import tensorflow as tf

from util.torch_to_tf import torch_mse_loss

import numpy as np

def test_mse_loss():

    # 示例数据
    input = torch.tensor([0.5, 1.0, 1.5])
    target = torch.tensor([1.0, 1.0, 1.0])

    
    loss1 = F.mse_loss(input, target, reduction='mean')
    print(loss1)    
    print( type(loss1) )  

    loss2 = F.mse_loss(input, target, reduction='sum')
    print(loss2)    
    print( type(loss2) )  

    loss3 = F.mse_loss(input, target, reduction='none')
    print(loss3)    
    print( type(loss3) )  



    input = tf.constant([0.5, 1.0, 1.5])
    target = tf.constant([1.0, 1.0, 1.0])


    loss_tf1 = torch_mse_loss(input, target, reduction='mean')
    print(loss_tf1)
    print( type(loss_tf1) )

    loss_tf2 = torch_mse_loss(input, target, reduction='sum')
    print(loss_tf2)
    print( type(loss_tf2) )

    loss_tf3 = torch_mse_loss(input, target, reduction='none')
    print(loss_tf3)
    print( type(loss_tf3) )


    assert np.allclose(loss_tf1.numpy(), loss1.numpy())
    assert np.allclose(loss_tf2.numpy(), loss2.numpy())
    assert np.allclose(loss_tf3.numpy(), loss3.numpy())




test_mse_loss()





