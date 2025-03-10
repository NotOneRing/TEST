import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch network
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# create a PyTorch tensor
tensor_pytorch = torch.tensor([1.0, 2.0], requires_grad=False)

# use the requires_grad_() method to set gradient calculation
tensor_pytorch.requires_grad_()  # set tensor_pytorch to have gradient calculation

# check if the tensor needs to have gradient calculation
print("PyTorch tensor requires_grad:", tensor_pytorch.requires_grad)

# create a PyTorch network and train
net = SimpleNet()
optimizer = optim.SGD(net.parameters(), lr=0.01)
# input_tensor = torch.tensor([1.0, 2.0], requires_grad=True)

input_tensor = torch.tensor([1.0, 2.0], requires_grad=False)

print("input_tensor.shape = ", input_tensor.shape)

output = net(input_tensor)

# backward propagation
output.backward()

# check gradients
print("PyTorch gradient for input_tensor:", input_tensor.grad)

# check gradients of the networks' parameters
for name, param in net.named_parameters():
    print(f"Gradient for {name}: {param.grad}")




import tensorflow as tf

# TensorFlow network
class SimpleNet_tf(tf.keras.Model):
    def __init__(self):
        super(SimpleNet_tf, self).__init__()
        self.dense1 = tf.keras.layers.Dense(2)
        self.dense2 = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# create a TensorFlow variable
tensor_tf = tf.Variable([1.0, 2.0], trainable=False)


# def torch_tensor_requires_grad_(tensor, requires_grad=True):
#     if requires_grad:
#         return tf.Variable(tensor, trainable=True)
#     else:
#         return tf.Variable(tensor, trainable=False)
    
from util.torch_to_tf import torch_tensor_requires_grad_

# use this wrapper to set the trainable property
# tensor_tf = torch_tensor_requires_grad_(tensor_tf, requires_grad=True)

# check if the tensor needs the gradient calculation
print("TensorFlow tensor trainable:", tensor_tf.trainable)

# create a TensorFlow network and train
net_tf = SimpleNet_tf()

import numpy as np

_ = net_tf(tf.constant(np.random.randn(1, 2).astype(np.float32)))


# print("net_tf.dense1.trainable_weights = ", net_tf.dense1.trainable_weights)

net_tf.dense1.trainable_weights[0].assign(net.fc1.weight.detach().numpy().T)  # kernel
net_tf.dense1.trainable_weights[1].assign(net.fc1.bias.detach().numpy())     # bias

net_tf.dense2.trainable_weights[0].assign(net.fc2.weight.detach().numpy().T)  # kernel
net_tf.dense2.trainable_weights[1].assign(net.fc2.bias.detach().numpy())     # bias


# input_tensor_tf = tf.Variable([1.0, 2.0], trainable=True)
input_tensor_tf = tf.Variable([1.0, 2.0], trainable=True)

input_tensor_tf = torch_tensor_requires_grad_(input_tensor_tf, requires_grad=False)


print("input_tensor_tf.shape = ", input_tensor_tf.shape)


with tf.GradientTape(persistent=True) as tape:
    input_tensor_tf = tf.reshape( input_tensor_tf, [1, 2] )
    output = net_tf(input_tensor_tf)

# grads = tape.gradient(output, net_tf.trainable_variables)

# print("TensorFlow gradient for trainable_variables:", grads)

grads0 = tape.gradient(output, input_tensor_tf)


grads1 = tape.gradient(output, net_tf.dense1.trainable_weights[0])
grads2 = tape.gradient(output, net_tf.dense1.trainable_weights[1])

grads3 = tape.gradient(output, net_tf.dense2.trainable_weights[0])
grads4 = tape.gradient(output, net_tf.dense2.trainable_weights[1])


print("TensorFlow gradient for input_tensor:", grads0)
print("TensorFlow gradient for fc1.weight:", grads1)
print("TensorFlow gradient for fc1.bias:", grads2)
print("TensorFlow gradient for fc2.weight:", grads3)
print("TensorFlow gradient for fc2.bias:", grads4)

if grads0:
    assert np.allclose(grads0.numpy(), input_tensor.grad.numpy())
else:
    assert grads0 == input_tensor.grad

tf_grads1 = grads1.numpy()
torch_grads1 =  dict(net.named_parameters())['fc1.weight'].grad.detach().numpy()

print("tf_grads1 = ", tf_grads1)
print("torch_grads1 = ", torch_grads1)

assert np.allclose(tf_grads1, torch_grads1.T)

assert np.allclose(grads2.numpy(), dict(net.named_parameters())['fc1.bias'].grad.detach().numpy())

assert np.allclose(grads3.numpy(), dict(net.named_parameters())['fc2.weight'].grad.detach().numpy().T)
assert np.allclose(grads4.numpy(), dict(net.named_parameters())['fc2.bias'].grad.detach().numpy())


del tape






















