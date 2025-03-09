import torch
import tensorflow as tf
import numpy as np


from util.torch_to_tf import torch_flip


# test case 1: generate a tensor of random integers within range (0, 10) with shape (2, 3, 4)
def test_case_1():
    input_array = np.random.randint(0, 10, (2, 3, 4))
    tf_input = tf.constant(input_array)
    torch_input = torch.tensor(input_array)

    dims = [0, 2]
    tf_output = torch_flip(tf_input, dims)
    torch_output = torch.flip(torch_input, dims)

    print("Test Case 1 - Shape (2, 3, 4), Flip dims: [0, 2]")
    print("Original Tensor (Torch):\n", torch_input)
    print("TensorFlow Output:\n", tf_output)
    print("PyTorch Output:\n", torch_output)
    print("Are the outputs the same? ", np.array_equal(tf_output.numpy(), torch_output.numpy()))
    print("-" * 50)
    assert np.array_equal(tf_output.numpy(), torch_output.numpy())



# test case 2: generate a tensor of random integers within range (0, 10) with shape (3, 3)
def test_case_2():
    input_array = np.random.randint(0, 10, (3, 3))
    tf_input = tf.constant(input_array)
    torch_input = torch.tensor(input_array)

    dims = [0]  # Flip along the first dimension
    tf_output = torch_flip(tf_input, dims)
    torch_output = torch.flip(torch_input, dims)

    print("Test Case 2 - Shape (3, 3), Flip dims: [0]")
    print("Original Tensor (Torch):\n", torch_input)
    print("TensorFlow Output:\n", tf_output)
    print("PyTorch Output:\n", torch_output)
    print("Are the outputs the same? ", np.array_equal(tf_output.numpy(), torch_output.numpy()))
    print("-" * 50)
    assert np.array_equal(tf_output.numpy(), torch_output.numpy())



# test case 3: generate a tensor of random integers within range (0, 10) with shape (1, 5, 2)
def test_case_3():
    input_array = np.random.randint(0, 10, (1, 5, 2))
    tf_input = tf.constant(input_array)
    torch_input = torch.tensor(input_array)

    dims = [1]  # Flip along the second dimension
    tf_output = torch_flip(tf_input, dims)
    torch_output = torch.flip(torch_input, dims)

    print("Test Case 3 - Shape (1, 5, 2), Flip dims: [1]")
    print("Original Tensor (Torch):\n", torch_input)
    print("TensorFlow Output:\n", tf_output)
    print("PyTorch Output:\n", torch_output)
    print("Are the outputs the same? ", np.array_equal(tf_output.numpy(), torch_output.numpy()))
    print("-" * 50)
    assert np.array_equal(tf_output.numpy(), torch_output.numpy())



# test case 4: generate a tensor of random integers within range (0, 10) with shape (4, 1, 3)
def test_case_4():
    input_array = np.random.randint(0, 10, (4, 1, 3))
    tf_input = tf.constant(input_array)
    torch_input = torch.tensor(input_array)

    dims = [2]  # Flip along the third dimension
    tf_output = torch_flip(tf_input, dims)
    torch_output = torch.flip(torch_input, dims)

    print("Test Case 4 - Shape (4, 1, 3), Flip dims: [2]")
    print("Original Tensor (Torch):\n", torch_input)
    print("TensorFlow Output:\n", tf_output)
    print("PyTorch Output:\n", torch_output)
    print("Are the outputs the same? ", np.array_equal(tf_output.numpy(), torch_output.numpy()))
    print("-" * 50)
    assert np.array_equal(tf_output.numpy(), torch_output.numpy())



# test case 5: generate a tensor of random integers within range (0, 10) with shape (3, 1, 5, 2)
def test_case_5():
    input_array = np.random.randint(0, 10, (3, 1, 5, 2))
    tf_input = tf.constant(input_array)
    torch_input = torch.tensor(input_array)

    dims = [0, 3]  # Flip along the first and last dimension
    tf_output = torch_flip(tf_input, dims)
    torch_output = torch.flip(torch_input, dims)

    print("Test Case 5 - Shape (3, 1, 5, 2), Flip dims: [0, 3]")
    print("Original Tensor (Torch):\n", torch_input)
    print("TensorFlow Output:\n", tf_output)
    print("PyTorch Output:\n", torch_output)
    print("Are the outputs the same? ", np.array_equal(tf_output.numpy(), torch_output.numpy()))
    print("-" * 50)
    assert np.array_equal(tf_output.numpy(), torch_output.numpy())



# run all test cases
def run_tests():
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()

run_tests()

