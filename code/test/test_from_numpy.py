import numpy as np
import torch
import tensorflow as tf

from util.torch_to_tf import torch_from_numpy

def test_from_numpy():
        
    # use NumPy to generate one array
    numpy_array = np.full((3, 3), 5.0)  # create a 3x3 array filled with 5.0

    # convert the NumPy array to the PyTorch tensor and the TensorFlow tensor
    torch_tensor = torch.from_numpy(numpy_array)
    tf_tensor = torch_from_numpy(numpy_array)


    assert np.allclose(torch_tensor.numpy(), tf_tensor.numpy())

test_from_numpy()




