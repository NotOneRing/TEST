import numpy as np
import torch
import tensorflow as tf

from util.torch_to_tf import torch_full

def test_full():
        
    # use NumPy to generate one array
    numpy_array = np.full((3, 3), 5.0)  # create an array of shape (3, 3), filled with 5.0

    # convert the NumPy array into the PyTorch tensor and the TensorFlow tensor
    torch_tensor = torch.tensor(numpy_array, dtype=torch.float32)
    tf_tensor = tf.convert_to_tensor(numpy_array, dtype=tf.float32)

    # use torch.full to create an array in PyTorch with the same shape and values as the NumPy array
    torch_result = torch.full((3, 3), 5.0, dtype=torch.float32)

    # use tf.fill to create an array in TensorFlow with the same shape and values as the NumPy array
    tf_result = torch_full([3, 3], 5.0)

    # print result
    print("Original NumPy array:\n", numpy_array)
    print("\nTorch tensor:\n", torch_tensor)
    print("\nTensorFlow tensor:\n", tf_tensor)
    print("\nTorch full tensor:\n", torch_result)
    print("\nTensorFlow fill tensor:\n", tf_result)

    assert np.allclose(torch_result.numpy(), tf_result.numpy())

test_full()




