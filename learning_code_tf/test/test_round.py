import torch
import tensorflow as tf

from util.torch_to_tf import torch_round

import numpy as np

def test_round():
    # 创建一个浮动类型的张量
    tensor = torch.tensor([1.1, 2.5, 3.7, -1.4])

    # 四舍五入
    rounded_tensor = torch.round(tensor)

    print(rounded_tensor)





    # 创建一个浮动类型的张量
    tensor = tf.constant([1.1, 2.5, 3.7, -1.4])

    # 四舍五入
    tf_rounded_tensor = torch_round(tensor)

    print(tf_rounded_tensor)


    assert np.allclose(rounded_tensor, tf_rounded_tensor)




test_round()

