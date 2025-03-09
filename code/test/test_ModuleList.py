import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

from util.torch_to_tf import nn_ModuleList


# initialize alignment function
def align_initialization(torch_modulelist, tf_modulelist):
    for torch_layer, tf_layer in zip(torch_modulelist, tf_modulelist):
        if isinstance(torch_layer, nn.Linear) and isinstance(tf_layer, tf.keras.layers.Dense):

            # print("torch_layer.shape = ", torch_layer.weight.detach().shape)
            # print("tf_layer.shape = ", tf_layer.kernel.shape)

            dummy_input = tf.random.normal([1, torch_layer.in_features])
            tf_layer(dummy_input)  # Trigger weight initialization

            # extract PyTorch's initialization weights and biases
            torch_weight = torch_layer.weight.detach().numpy()
            torch_bias = torch_layer.bias.detach().numpy() if torch_layer.bias is not None else None
            
            # assign to the TensorFlow layer
            tf_layer.kernel.assign(torch_weight.T)  # transpose the weights
            if torch_bias is not None:
                tf_layer.bias.assign(torch_bias)
            else:
                print("torch_bias is None")



# PyTorch testing function
def test_torch_modulelist(module_list, input_tensor):


    outputs = []
    for module in module_list:
        input_tensor = module(input_tensor)
        outputs.append(input_tensor)

    return outputs



# TensorFlow testing function
def test_tf_modulelist(module_list, input_tensor):


    outputs = module_list(input_tensor)
    return outputs



# compare testing results
def test_results():
    # create the same input tensor
    torch_input = torch.randn(1, 10)
    tf_input = tf.convert_to_tensor(torch_input.detach().numpy(), dtype=tf.float32)
    
    # # get PyTorch's and TensorFlow's models and outputs
    # torch_modulelist, torch_outputs = test_torch_modulelist(torch_input)
    # tf_modulelist, tf_outputs = test_tf_modulelist(tf_input)
    
    torch_modulelist = nn.ModuleList([
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ])

    tf_modulelist = nn_ModuleList([
        tf.keras.layers.Dense(20, activation=None),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(5, activation=None)
    ])

    # align initialization
    align_initialization(torch_modulelist, tf_modulelist)
    

    # # get PyTorch's and TensorFlow's models and outputs
    torch_outputs = test_torch_modulelist(torch_modulelist, torch_input)
    tf_outputs = test_tf_modulelist(tf_modulelist, tf_input)
    

    # ensure the number of output layers are consistent
    assert len(torch_outputs) == len(tf_outputs), \
        f"Mismatch in number of layers: {len(torch_outputs)} (torch) vs {len(tf_outputs)} (tf)"
    
    # compare the output of each layer
    for i, (torch_out, tf_out) in enumerate(zip(torch_outputs, tf_outputs)):
        # convert to the NumPy arrays
        torch_out_np = torch_out.detach().numpy()
        tf_out_np = tf_out.numpy()
        
        # check both shape are consistent
        if torch_out_np.shape != tf_out_np.shape:
            print(f"Layer {i} shape mismatch: Torch {torch_out_np.shape}, TF {tf_out_np.shape}")
        else:
            print(f"Layer {i} shape match: {torch_out_np.shape}")
        
        # check both values are consistent
        match = np.allclose(torch_out_np, tf_out_np, atol=1e-5)
        print(f"Layer {i} output match: {match}")

        assert match
        
        # if not match, print the difference
        if not match:
            diff = np.abs(torch_out_np - tf_out_np)
            print(f"Layer {i} max difference: {np.max(diff)}")
            print(f"Layer {i} difference matrix:\n{diff}")
        
        # print output values of each layer(optional)
        print(f"Layer {i} Torch output:\n{torch_out_np}")
        print(f"Layer {i} TF output:\n{tf_out_np}")
        print("-" * 50)



# run test
test_results()
