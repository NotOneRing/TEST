import torch
import tensorflow as tf
import numpy as np

# # TensorFlow version of torch_rand
# def torch_rand(*size, dtype=tf.dtypes.float32):
#     return tf.random.uniform(shape=size, dtype=dtype)
from util.torch_to_tf import torch_rand

# Test function to compare PyTorch and TensorFlow outputs
def test_random_functions():
    # Define shape for random tensor
    shape = (3, 4)

    # PyTorch: torch.rand
    pytorch_tensor = torch.rand(*shape)
    print("PyTorch Tensor (torch.rand):")
    print(pytorch_tensor)
    
    # TensorFlow: torch_rand (wrapper for tf.random.uniform)
    tensorflow_tensor = torch_rand(*shape)
    print("\nTensorFlow Tensor (torch_rand wrapper):")
    print(tensorflow_tensor)

    # Check if the values are approximately equal
    same_values = np.allclose(pytorch_tensor.numpy().shape, tensorflow_tensor.numpy().shape, atol=1e-5)
    print("\nAre the PyTorch and TensorFlow outputs the same shape?", same_values)



    # PyTorch: torch.rand
    pytorch_tensor = torch.rand(shape)
    print("PyTorch Tensor (torch.rand):")
    print(pytorch_tensor)
    
    # TensorFlow: torch_rand (wrapper for tf.random.uniform)
    tensorflow_tensor = torch_rand(shape)
    print("\nTensorFlow Tensor (torch_rand wrapper):")
    print(tensorflow_tensor)

    # Check if the values are approximately equal
    same_values = np.allclose(pytorch_tensor.numpy().shape, tensorflow_tensor.numpy().shape, atol=1e-5)
    print("\nAre the PyTorch and TensorFlow outputs the same shape?", same_values)


# Run the test
test_random_functions()
