import tensorflow as tf
from util.torch_to_tf import torch_dot

def test_torch_dot():

    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.constant([4.0, 5.0, 6.0])

    dot_product = torch_dot(a, b)

    print(dot_product)


    import torch

    a_torch = torch.tensor(a.numpy())
    b_torch = torch.tensor(b.numpy())


    torch_dot_product = torch.dot(a_torch, b_torch)

    print(torch_dot_product)


    assert dot_product.numpy() == torch_dot_product.numpy(), "The tensorflow output is not equivalent to the torch one"


test_torch_dot()




