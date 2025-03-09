import torch
import tensorflow as tf
import numpy as np


from util.torch_to_tf import torch_tensor_expand_as


def test_expand_as():
    # test examples list
    test_cases = [
        {"input_shape": (1,), "target_shape": (3,)},       # 1D -> 1D
        {"input_shape": (1, 4), "target_shape": (3, 4)},   # 2D -> 2D
        {"input_shape": (2, 1), "target_shape": (2, 3)},   # 2D -> 2D (broadcastable along last dimension)
        {"input_shape": (1, 1, 4), "target_shape": (2, 3, 4)},  # 3D -> 3D
        {"input_shape": (4,), "target_shape": (2, 3, 4)},  # 1D -> 3D (broadcast along first two dimensions)
    ]
    
    for i, case in enumerate(test_cases):
        input_shape = case["input_shape"]
        target_shape = case["target_shape"]
        
        # create test data
        input_torch = torch.rand(input_shape)
        target_torch = torch.rand(target_shape)
        
        input_tf = tf.convert_to_tensor(input_torch.numpy())
        target_tf = tf.convert_to_tensor(target_torch.numpy())
        
        # PyTorch's expand_as
        expanded_torch = input_torch.expand_as(target_torch)
        
        # TensorFlow's broadcast_to as torch_tensor_expand_as
        expanded_tf = torch_tensor_expand_as(input_tf, target_tf)
        
        # use NumPy to compare
        expanded_torch_np = expanded_torch.numpy()
        expanded_tf_np = expanded_tf.numpy()
        
        # print test results
        print(f"Test case {i + 1}:")
        print(f"Input shape: {input_shape}, Target shape: {target_shape}")
        print(f"PyTorch expanded result:\n{expanded_torch_np}")
        print(f"TensorFlow expanded result:\n{expanded_tf_np}")
        
        # check if outputs are equivalent
        assert np.array_equal(expanded_torch_np, expanded_tf_np), f"Test case {i + 1} failed!"
        print(f"Test case {i + 1} passed!\n{'-' * 40}\n")

# run testing
test_expand_as()
