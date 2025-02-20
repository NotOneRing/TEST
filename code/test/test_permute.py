
from util.torch_to_tf import torch_tensor_permute

import torch
import tensorflow as tf
import numpy as np

def test_comparison():

    input_tensor_torch = torch.randn(1, 2, 3)
    
    output_tensor_torch = input_tensor_torch.permute(1, 2, 0)
    
    print(output_tensor_torch.shape)
    

    input_tensor_tf = tf.convert_to_tensor(input_tensor_torch.numpy())
    
    # output_tensor_tf = tf.transpose(input_tensor_tf, perm=[1, 2, 0])
    # output_tensor_tf = torch_tensor_permute(input_tensor_tf, 1, 2, 0)
    output_tensor_tf = torch_tensor_permute(input_tensor_tf, (1, 2, 0) )
    
    print(output_tensor_tf.shape)

    # print("output_tensor_torch = ", output_tensor_torch)

    # print("output_tensor_tf = ", output_tensor_tf)


    assert np.allclose(output_tensor_torch.numpy(), output_tensor_tf.numpy())



test_comparison()




