

import torch


import tensorflow as tf

from util.torch_to_tf import torch_randperm

from util.config import DEBUG

def test_randperm():

    # create an integer sequence from 0 to 9 and shuffle it randomly
    tensor = torch.randperm(10)
    print(tensor)



    # create an integer sequence from 0 to 9 and shuffle it randomly
    tensor_tf = torch_randperm(10)
    print("type(tensor_tf)", type(tensor_tf))
    print(tensor_tf)



test_randperm()