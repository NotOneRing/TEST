def test_cumprod():
    import torch

    # create a tensor of 1 dimension
    tensor = torch.tensor([1, 2, 3, 4])

    # calculate cumulative products
    cumprod_result = torch.cumprod(tensor, dim=0)
    print(cumprod_result)



    tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # calculate the cumulative product along the zeroth dimension
    cumprod_dim_0 = torch.cumprod(tensor_2d, dim=0)
    print("cumprod along dim 0:")
    print(cumprod_dim_0)

    # calculate the cumulative product along the first dimension
    cumprod_dim_1 = torch.cumprod(tensor_2d, dim=1)
    print("cumprod along dim 1:")
    print(cumprod_dim_1)



    from util.torch_to_tf import torch_cumprod

    import tensorflow as tf

    # create a tensor of 1 dimension
    tensor = tf.constant([1, 2, 3, 4])

    # calculate cumulative products
    tf_cumprod_result = torch_cumprod(tensor, dim=0)
    print(cumprod_result)



    tensor_2d = tf.constant([[1, 2, 3], [4, 5, 6]])

    # calculate the cumulative product along the zeroth dimension
    tf_cumprod_dim_0 = torch_cumprod(tensor_2d, dim=0)
    print("cumprod along axis 0:")
    print(cumprod_dim_0)

    # calculate the cumulative product along the first dimension
    tf_cumprod_dim_1 = torch_cumprod(tensor_2d, dim=1)
    print("cumprod along axis 1:")
    print(cumprod_dim_1)


    import numpy as np

    assert np.array_equal(cumprod_result.numpy(), tf_cumprod_result.numpy())

    assert np.array_equal(cumprod_dim_0.numpy(), tf_cumprod_dim_0.numpy())

    assert np.array_equal(cumprod_dim_1.numpy(), tf_cumprod_dim_1.numpy())


test_cumprod()
