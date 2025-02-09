import torch
import tensorflow as tf
import numpy as np





# 固定种子以确保相同的随机数
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


from util.torch_to_tf import torch_nn_functional_grid_sample
# , torch_nn_functional_grid_sample2


def test_case1_align_corners():
    input_tensor_torch = torch.range(start = 1, end = 1*3*3*3).reshape(1, 3, 3, 3)

    input_tensor_torch = input_tensor_torch.permute(0, 3, 1, 2)  # Convert to NCHW format for PyTorch

    input_tensor_tf = tf.convert_to_tensor(input_tensor_torch.numpy())

    grid_torch = torch.range(start = 1, end = 1*3*3*2).reshape(1, 3, 3, 2)

    # grid_torch = (grid_torch - 1*32*32*2  / 2) / (1*32*32*2)
    # grid_torch = (grid_torch - 1*3*3*2  / 2) / (1*3*3*2)
    grid_torch = (grid_torch * 2 - 1*3*3*2) / (1*3*3*2)

    grid_tf = tf.convert_to_tensor(grid_torch.numpy())

    output_torch = torch.nn.functional.grid_sample(input_tensor_torch, grid_torch, align_corners = True)

    # from util.torch_to_tf import torch_tensor_clone
    # input_tensor_tf_copy = torch_tensor_clone(input_tensor_tf)
    # output_tf = torch_nn_functional_grid_sample2(input_tensor_tf, grid_tf, align_corners = True)

    output_tf = torch_nn_functional_grid_sample(input_tensor_tf, grid_tf, align_corners = True)

    torch_output = output_torch.detach().numpy()
    tensorflow_output = output_tf.numpy()

    print("torch_output = ", torch_output)
    print("tensorflow_output = ", tensorflow_output)


    assert np.allclose(torch_output, tensorflow_output, atol=1e-5)





def test_case2_align_corners():
    input_tensor_torch = torch.range(start = 1, end = 1*3*3*3).reshape(1, 3, 3, 3)

    input_tensor_torch = input_tensor_torch.permute(0, 3, 1, 2)  # Convert to NCHW format for PyTorch

    input_tensor_tf = tf.convert_to_tensor(input_tensor_torch.numpy())

    grid_torch = torch.range(start = 1, end = 1*3*3*2).reshape(1, 3, 3, 2)

    # grid_torch = (grid_torch - 1*32*32*2  / 2) / (1*32*32*2)
    # grid_torch = (grid_torch - 1*3*3*2  / 2) / (1*3*3*2)
    grid_torch = (grid_torch * 2 - 1*3*3*2) / (1*3*3*2)

    grid_tf = tf.convert_to_tensor(grid_torch.numpy())

    output_torch = torch.nn.functional.grid_sample(input_tensor_torch, grid_torch, align_corners = False)

    output_tf = torch_nn_functional_grid_sample(input_tensor_tf, grid_tf, align_corners = False)

    torch_output = output_torch.detach().numpy()
    tensorflow_output = output_tf.numpy()

    # print("torch_output = ", torch_output)
    # print("tensorflow_output = ", tensorflow_output)

    assert np.allclose(torch_output, tensorflow_output, atol=1e-5)




def test_case3():
    input_tensor_torch = torch.range(start = 1, end = 1*3*3*3).reshape(1, 3, 3, 3)

    input_tensor_torch = input_tensor_torch.permute(0, 3, 1, 2)  # Convert to NCHW format for PyTorch

    input_tensor_tf = tf.convert_to_tensor(input_tensor_torch.numpy())

    grid_torch = torch.range(start = 1, end = 1*3*3*2).reshape(1, 3, 3, 2)

    # grid_torch = (grid_torch - 1*32*32*2  / 2) / (1*32*32*2)
    grid_torch = (grid_torch - 1*3*3*2  / 2) / (1*3*3*2)
    # grid_torch = (grid_torch * 2 - 1*3*3*2) / (1*3*3*2)

    grid_tf = tf.convert_to_tensor(grid_torch.numpy())

    output_torch = torch.nn.functional.grid_sample(input_tensor_torch, grid_torch, align_corners = True)

    output_tf = torch_nn_functional_grid_sample(input_tensor_tf, grid_tf, align_corners = True)

    torch_output = output_torch.detach().numpy()
    tensorflow_output = output_tf.numpy()

    # print("torch_output = ", torch_output)
    # print("tensorflow_output = ", tensorflow_output)

    assert np.allclose(torch_output, tensorflow_output, atol=1e-5)



test_case1_align_corners()

test_case2_align_corners()

test_case3()








