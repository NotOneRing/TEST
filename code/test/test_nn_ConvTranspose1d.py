import tensorflow as tf

from util.torch_to_tf import nn_ConvTranspose1d


import torch
import numpy as np

# PyTorch ConvTranspose1d
torch_layer = torch.nn.ConvTranspose1d(3, 3, 4, 2, 1)  # padding=1
torch_input = torch.randn(1, 3, 10)
torch_output = torch_layer(torch_input).detach().numpy()


# TensorFlow ConvTranspose1d
tf_layer = nn_ConvTranspose1d(3, 3, 4, 2, 1)  # only implement the padding=1 case
tf_input = tf.convert_to_tensor(torch_input.numpy(), dtype=tf.float32)
tf_output = tf_layer(tf_input).numpy()

# same weights
torch_weights = torch_layer.weight.detach().numpy()  # PyTorch weight
torch_weights_tf = np.transpose(torch_weights, (2, 1, 0))  # convert to the shape of the TensorFlow shape
tf_layer.conv1d_transpose.kernel.assign(torch_weights_tf)

# same bias
if torch_layer.bias is not None:
    tf_layer.conv1d_transpose.bias.assign(torch_layer.bias.detach().numpy())

tf_output = tf_layer(tf_input).numpy()


print("max difference:", np.abs(torch_output - tf_output).max())
print("PyTorch output shape:", torch_output.shape)
print("TensorFlow output shape:", tf_output.shape)


