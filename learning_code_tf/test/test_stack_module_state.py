import torch
import torch.nn as nn

import numpy as np

from torch.func import stack_module_state

# 定义简单模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)



temp = [SimpleNet()]

stacked_params1, stacked_buffers1 = stack_module_state(temp)


result1 = []

for k, v in stacked_params1.items():
    result1.append(v.shape)


print("Stacked Parameters Shape:", {k: v.shape for k, v in stacked_params1.items()})
print("Stacked Buffers:", stacked_buffers1)


# 创建多个模型实例
models = [SimpleNet() for _ in range(3)]

# 堆叠这些模型的参数和缓冲区
stacked_params2, stacked_buffers2 = stack_module_state(models)

print("Stacked Parameters Shape:", {k: v.shape for k, v in stacked_params2.items()})
print("Stacked Buffers:", stacked_buffers2)


result2 = []

for k, v in stacked_params2.items():
    result2.append(v.shape)



# W的维度是(out_features, in_features)，因为y = xW^{\top} + b

import tensorflow as tf

from util.torch_to_tf import nn_Linear, torch_func_stack_module_state

class tf_SimpleNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc = nn_Linear(2, 1)

    def call(self, x):
        return self.fc(x)

tf_result1 = []

temp = [tf_SimpleNet()]

for network in temp:
    _ = network(tf.constant(np.random.randn(1, 2).astype(np.float32)))


tf_stacked_params1, tf_stacked_buffers1 = torch_func_stack_module_state(temp)

print("Stacked Parameters Shape:", {k: v.shape for k, v in tf_stacked_params1.items()})
print("Stacked Buffers:", tf_stacked_buffers1)

for k, v in tf_stacked_params1.items():
    tf_result1.append(v.shape)

# 创建多个模型实例
models = [tf_SimpleNet() for _ in range(3)]

for network in models:
    _ = network(tf.constant(np.random.randn(1, 2).astype(np.float32)))

# 堆叠这些模型的参数和缓冲区
tf_stacked_params2, tf_stacked_buffers2 = torch_func_stack_module_state(models)

print("Stacked Parameters Shape:", {k: v.shape for k, v in tf_stacked_params2.items()})
print("Stacked Buffers:", tf_stacked_buffers2)


tf_result2 = []

for k, v in tf_stacked_params2.items():
    tf_result2.append(v.shape)


import numpy as np

def test_stack_module_state():

    for i in range(len(result1)):
        assert np.allclose(result1[i], tf_result1[i])

    for i in range(len(result2)):
        assert np.allclose(result2[i], tf_result2[i])


test_stack_module_state()































