import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch 网络
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 创建一个 PyTorch tensor
tensor_pytorch = torch.tensor([1.0, 2.0], requires_grad=False)

# 使用 requires_grad_() 方法设置梯度计算
tensor_pytorch.requires_grad_()  # 设置为需要梯度计算

# 查看 tensor 是否需要梯度
print("PyTorch tensor requires_grad:", tensor_pytorch.requires_grad)

# 创建 PyTorch 网络并训练
net = SimpleNet()
optimizer = optim.SGD(net.parameters(), lr=0.01)
# input_tensor = torch.tensor([1.0, 2.0], requires_grad=True)

input_tensor = torch.tensor([1.0, 2.0], requires_grad=False)

print("input_tensor.shape = ", input_tensor.shape)

output = net(input_tensor)

# 反向传播
output.backward()

# 查看梯度
print("PyTorch gradient for input_tensor:", input_tensor.grad)

# 查看网络中参数的梯度
for name, param in net.named_parameters():
    print(f"Gradient for {name}: {param.grad}")




import tensorflow as tf

# TensorFlow 网络
class SimpleNet_tf(tf.keras.Model):
    def __init__(self):
        super(SimpleNet_tf, self).__init__()
        self.dense1 = tf.keras.layers.Dense(2)
        self.dense2 = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# 创建一个 TensorFlow 变量
tensor_tf = tf.Variable([1.0, 2.0], trainable=False)


def torch_tensor_requires_grad_(tensor, requires_grad=True):
    if requires_grad:
        return tf.Variable(tensor, trainable=True)
    else:
        return tf.Variable(tensor, trainable=False)
    


# 使用该 wrapper 设置 trainable 属性
# tensor_tf = torch_tensor_requires_grad_(tensor_tf, requires_grad=True)

# 查看 tensor 是否需要梯度
print("TensorFlow tensor trainable:", tensor_tf.trainable)

# 创建 TensorFlow 网络并训练
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

grads = tape.gradient(output, input_tensor_tf)

print("TensorFlow gradient for input_tensor:", grads)


del tape




















