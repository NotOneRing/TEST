import torch
import tensorflow as tf
import numpy as np


from util.torch_to_tf import torch_split





def test_split_func():


    # Test case 1: Split a tensor into equal parts
    print("Test case 1: Split a tensor into equal parts")

    # PyTorch tensor
    pytorch_tensor = torch.randn(6, 4)  # Shape: (6, 4)
    pytorch_split = torch.split(pytorch_tensor, 4, dim=0)  # Split into 3 parts along dim 0 (6 elements, 2 per part)

    print("type(pytorch_split) = ", type(pytorch_split))
    print("len(pytorch_split) = ", len(pytorch_split))

    # TensorFlow tensor
    tf_tensor = tf.convert_to_tensor( pytorch_tensor.numpy() )
    tf_split = torch_split(tf_tensor, 4, dim=0)  # Split into 3 parts along axis 0

    print("type(tf_split) = ", type(tf_split))
    print("len(tf_split) = ", len(tf_split))

    # Compare results for Test case 1
    print("Comparing Test case 1 results:")

    for i, (pytorch_part, tf_part) in enumerate(zip(pytorch_split, tf_split)):
        print(f"\nPart {i + 1}:")
        print("PyTorch value:")
        print(pytorch_part.detach().numpy())  # Convert to numpy for comparison
        print("TensorFlow value:")
        print(tf_part.numpy())

        assert np.allclose(pytorch_part.detach().numpy(), tf_part.numpy(), atol=1e-5)













    # Test case 2: Split tensor with different sizes for each part
    print("\nTest case 2: Split a tensor with different sizes for each part")

    # PyTorch tensor
    pytorch_tensor = torch.randn(10, 4)  # Shape: (10, 4)
    pytorch_split = torch.split(pytorch_tensor, [3, 3, 4], dim=0)  # Split into parts of size 3, 3, and 4 along dim 0

    # TensorFlow tensor
    tf_tensor = tf.convert_to_tensor( pytorch_tensor.numpy() )

    tf_split = torch_split(tf_tensor, [3, 3, 4], dim=0)  # Split into parts of size 3, 3, and 4 along axis 0

    # Compare results for Test case 2
    print("Comparing Test case 2 results:")

    for i, (pytorch_part, tf_part) in enumerate(zip(pytorch_split, tf_split)):
        print(f"\nPart {i + 1}:")
        print("PyTorch value:")
        print(pytorch_part.detach().numpy())  # Convert to numpy for comparison
        print("TensorFlow value:")
        print(tf_part.numpy())

        assert np.allclose(pytorch_part.detach().numpy(), tf_part.numpy(), atol=1e-5)













    # Test case 3: Split tensor along another dimension (dim=1)
    print("\nTest case 3: Split a tensor along dimension 1")

    # PyTorch tensor
    pytorch_tensor = torch.randn(6, 4)  # Shape: (6, 4)
    pytorch_split = torch.split(pytorch_tensor, 2, dim=1)  # Split into 2 parts along dim 1 (4 elements, 2 per part)

    # TensorFlow tensor
    tf_tensor = tf.convert_to_tensor( pytorch_tensor.numpy() )

    tf_split = torch_split(tf_tensor, 2, dim=1)  # Split into 2 parts along axis 1

    # Compare results for Test case 3
    print("Comparing Test case 3 results:")

    for i, (pytorch_part, tf_part) in enumerate(zip(pytorch_split, tf_split)):
        print(f"\nPart {i + 1}:")
        print("PyTorch value:")
        print(pytorch_part.detach().numpy())  # Convert to numpy for comparison
        print("TensorFlow value:")
        print(tf_part.numpy())

        assert np.allclose(pytorch_part.detach().numpy(), tf_part.numpy(), atol=1e-5)








test_split_func()









