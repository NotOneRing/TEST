import torch


import tensorflow as tf

from util.torch_to_tf import torch_prod

def test_prod():
    # create a tensor
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

    result0 = torch.prod(tensor)
    print(result0)

    # calculate the prod along the dimension 0
    result1 = torch.prod(tensor, dim=0)
    print(result1)

    # calculate the prod along the dimension 1
    result2 = torch.prod(tensor, dim=1)
    print(result2)


    # create a tensor
    tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

    result0 = torch_prod(tensor)
    print(result0)

    # calculate the prod along the dimension 0
    result1 = torch_prod(tensor, dim=0)
    print(result1)

    # calculate the prod along the dimension 1
    result2 = torch_prod(tensor, dim=1)
    print(result2)

test_prod()



