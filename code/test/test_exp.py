

import torch
import tensorflow as tf
import numpy as np

def test_exp():

    input_data = np.random.randn(3, 4).astype(np.float32) 


    input_tensor_torch = torch.tensor(input_data)  
    output_torch = torch.exp(input_tensor_torch) 

    input_tensor_tf = tf.convert_to_tensor(input_data)  

    from util.torch_to_tf import torch_exp
    output_tf = torch_exp(input_tensor_tf)


    print("Input data:\n", input_data)
    print("\nPyTorch exp output:\n", output_torch.numpy())  

    print("\nTensorFlow exp output:\n", output_tf.numpy())  

    difference = np.abs(output_torch.numpy() - output_tf.numpy())
    print("\nDifference between PyTorch and TensorFlow outputs:\n", difference)

    assert np.allclose(output_torch.numpy(), output_tf.numpy())


test_exp()


