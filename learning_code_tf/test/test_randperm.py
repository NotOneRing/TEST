

import torch


import tensorflow as tf

from util.torch_to_tf import torch_randperm

from util.config import DEBUG

def test_randperm():

    # 生成一个从 0 到 9 的整数序列并随机打乱
    tensor = torch.randperm(10)
    print(tensor)



    # 生成一个从 0 到 9 的整数序列并随机打乱
    tensor_tf = torch_randperm(10)
    print("type(tensor_tf)", type(tensor_tf))
    print(tensor_tf)



test_randperm()