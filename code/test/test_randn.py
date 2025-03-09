import torch


def test_randn():
    # create 2x3 standard normal distribution random tensor
    tensor = torch.randn(2, 3)
    print(tensor)

    tensor = torch.randn([2, 3])
    print(tensor)



    import tensorflow as tf

    # create 2x3 standard normal distribution random tensor
    # tensor_tf = tf.random.normal([2, 3])

    from util.torch_to_tf import torch_randn

    tensor_tf = torch_randn(2, 3)

    print(tensor_tf)


    tensor_tf = torch_randn([2, 3])

    print(tensor_tf)

test_randn()



