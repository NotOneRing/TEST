import numpy as np
import torch
import tensorflow as tf
from util.torch_to_tf import nn_Conv1d

# set random seeds
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# parameters setting
batch_size = 2
in_channels = 4
out_channels = 8
length = 10
kernel_size = 3
stride = 1
padding = 1
dilation = 1
groups = 1  # ensure in_channels % groups == 0





# case2:
in_channels = 7
out_channels = 64
kernel_size = 1
batch_size = 256
in_channels = 7
length = 4
stride =  1
padding=0


# generate the input data (PyTorch format: batch_size, in_channels, length)
input_np = np.random.randn(batch_size, in_channels, length).astype(np.float32)
input_torch = torch.from_numpy(input_np)

# input_tf = tf.convert_to_tensor(np.transpose(input_np, (0, 2, 1)))  # convert to the TF format: (batch, length, channels)
input_tf = tf.convert_to_tensor(input_np)  # convert to the TF format: (batch, length, channels)


# create a PyTorch layer
torch_conv1d = torch.nn.Conv1d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    groups=groups,
    bias=True
)

output_torch = torch_conv1d(input_torch).detach().numpy()  # (batch, out_channels, length)

print("output_torch = ", output_torch)

print("output_torch.shape = ", output_torch.shape)



# create a TensorFlow layer
tf_conv1d = nn_Conv1d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    groups=groups,
    bias=True
)



# # construct a TF layer manually (input shape does not contain batch_size)
# input_shape = (length, in_channels)  # correct shape should be (length, in_channels)
# # tf_conv1d.build(input_shape=input_shape)

output_tf = tf_conv1d(input_tf).numpy()  

# copy weights and bias from PyTorch to TensorFlow
with torch.no_grad():
    # adjust weights' shape
    torch_weight = torch_conv1d.weight.numpy().transpose(2, 1, 0)  
    # pytorch is (out_channels, in_channels, kernel_size)
    # tensorflow is (kernel_size, in_channels, out_channels)
    torch_bias = torch_conv1d.bias.numpy()
    
    # check if the weights of the TF layer are already initialized
    if len(tf_conv1d.conv1d.weights) == 0:
        raise RuntimeError("Conv1D not initialized, please check the version of the TensorFlow or the shape of the input")
    
    # assign weights and biases
    tf_conv1d.conv1d.kernel.assign(torch_weight)
    tf_conv1d.conv1d.bias.assign(torch_bias)

# forward pass


output_tf = tf_conv1d(input_tf).numpy()  
# (batch, length, out_channels)

# output_tf = np.transpose(output_tf, (0, 2, 1))  # convert to (batch, out_channels, length)

# check consistency
if np.allclose(output_torch, output_tf, atol=1e-5):
    print("Outputs are consistent!")
else:
    print("Outputs are NOT consistent!")



















