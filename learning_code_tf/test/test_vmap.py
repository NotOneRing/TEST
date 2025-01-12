import torch

import tensorflow as tf

from util.torch_to_tf import torch_vmap

def outer_product(a):
  return tf.tensordot(a, a, 0)

batch_size = 100
a = tf.ones((batch_size, 32, 32))
c = tf.vectorized_map(outer_product, a)
assert c.shape == (batch_size, 32, 32, 32, 32)






def test_vmap1():
    # vmap() can also be nested, producing an output with multiple batched dimensions
    # torch.dot                            # [D], [D] -> []
    batched_dot = torch.vmap(torch.vmap(torch.dot))  # [N1, N0, D], [N1, N0, D] -> [N1, N0]
    x1, y1 = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
    print("x1 = ", x1)
    print("y1 = ", y1)


    print("x1.shape = ", x1.shape)
    print("y1.shape = ", y1.shape)


    result1 = batched_dot(x1, y1) # tensor of size [2, 3]

    print("result1 = ", result1)

    print("result1.shape = ", result1.shape)



    from util.torch_to_tf import torch_dot

    x1_tf = tf.convert_to_tensor(x1.numpy())
    y1_tf = tf.convert_to_tensor(y1.numpy())

    outputs_tf = torch_vmap( torch_vmap(torch_dot, x1_tf, y1_tf) )

    print("Outputs:", outputs_tf)



# def test_vmap2():
    
#     # torch.dot                            # [N], [N] -> []
#     batched_dot = torch.vmap(torch.dot, in_dims=1)  # [N, D], [N, D] -> [D]
#     x2, y2 = torch.randn(2, 5), torch.randn(2, 5)

#     print("x2.shape = ", x2.shape)
#     print("y2.shape = ", y2.shape)


#     print("x2 = ", x2)
#     print("y2 = ", y2)


#     result2 = batched_dot(x2, y2)   # output is [5] instead of [2] if batched along the 0th dimension

#     print("result2 = ", result2)
#     print("result2.shape = ", result2.shape)



def test_vmap3():
    f = lambda x: x ** 2
    x = torch.randn(2, 5)
    batched_pow = torch.vmap(f, out_dims=1)
    result = batched_pow(x) # [5, 2]
    print("x = ", x)
    print("result = ", result)

    f = lambda x: x ** 2
    x_tf = torch.tensor( x.numpy() )
    batched_pow = torch_vmap(f, out_dims=1)
    result = batched_pow(x) # [5, 2]

    print("x_tf = ", x_tf)
    print("result = ", result)


test_vmap1()


# test_vmap2()


test_vmap3()










