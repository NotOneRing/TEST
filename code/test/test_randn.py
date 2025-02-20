import torch


def test_randn():
    # 生成一个 2x3 的标准正态分布随机张量
    tensor = torch.randn(2, 3)
    print(tensor)

    tensor = torch.randn([2, 3])
    print(tensor)



    import tensorflow as tf

    # # 生成一个 2x3 的标准正态分布随机张量
    # tensor_tf = tf.random.normal([2, 3])

    from util.torch_to_tf import torch_randn

    tensor_tf = torch_randn(2, 3)

    print(tensor_tf)


    tensor_tf = torch_randn([2, 3])

    print(tensor_tf)

test_randn()



