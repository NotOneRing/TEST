import torch
import torch.nn as nn

import numpy as np

from torch.func import stack_module_state











# define the simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)



temp = [SimpleNet()]

stacked_params1, stacked_buffers1 = stack_module_state(temp)



result1 = []

for k, v in stacked_params1.items():
    # print("type(v) = ", type(v))
    # print("v = ", v)
    result1.append(v)

# print("type(stacked_params1) = ", type(stacked_params1))
# print("type(stacked_buffers1) = ", type(stacked_buffers1))


print("Stacked Parameters Shape:", {k: v.shape for k, v in stacked_params1.items()})
print("Stacked Buffers:", stacked_buffers1)


# create multiple model instances
models = [SimpleNet() for _ in range(3)]

# stack the parameters and buffers of these models
stacked_params2, stacked_buffers2 = stack_module_state(models)

print("Stacked Parameters Shape:", {k: v.shape for k, v in stacked_params2.items()})
print("Stacked Buffers:", stacked_buffers2)


result2 = []

for k, v in stacked_params2.items():
    result2.append(v)



# W's dimension is (out_features, in_features), because y = xW^{\top} + b

import tensorflow as tf

from util.torch_to_tf import nn_Linear, torch_func_stack_module_state

class tf_SimpleNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc = nn_Linear(2, 1)

    def call(self, x):
        return self.fc(x)


# temp_dense = nn_Linear(2, 1)
# print("temp_dense.trainable_variables = ", temp_dense.trainable_variables)

# print( "type(temp_dense.trainable_variables) = ", type(temp_dense.trainable_variables) )

# print("temp_dense.non_trainable_variables = ", temp_dense.non_trainable_variables)

# print( "type(temp_dense.non_trainable_variables) = ", type(temp_dense.non_trainable_variables) )

# _ = temp_dense(tf.constant(np.random.randn(1, 2).astype(np.float32)))

# print("temp_dense.trainable_variables = ", temp_dense.trainable_variables)

# print( "type(temp_dense.trainable_variables) = ", type(temp_dense.trainable_variables) )

# print("temp_dense.non_trainable_variables = ", temp_dense.non_trainable_variables)

# print( "type(temp_dense.non_trainable_variables) = ", type(temp_dense.non_trainable_variables) )


# for i in temp_dense.trainable_variables:
#     print("i = ", i)
#     print("type(i) = ", type(i))

# for var in temp_dense.trainable_variables:
#     print(f"Variable Name: {var.name}")
#     print(f"Variable Shape: {var.shape}")
#     print(f"Variable Type: {type(var)}")
#     print(f"Variable Value (as NumPy array): {var.numpy()}")
#     print("=" * 40)




tf_result1 = []

temp_tf = [tf_SimpleNet()]

for i, network in enumerate(temp_tf):
    _ = network(tf.constant(np.random.randn(1, 2).astype(np.float32)))

    # for torch_layer, tf_layer in zip(temp.fc, temp_tf.fc):
    if isinstance(temp[0].fc, nn.Linear):
        network.fc.trainable_weights[0].assign(temp[i].fc.weight.detach().numpy().T)  # kernel
        network.fc.trainable_weights[1].assign(temp[i].fc.bias.detach().numpy())     # bias



tf_stacked_params1, tf_stacked_buffers1 = torch_func_stack_module_state(temp_tf)

print("Stacked Parameters Shape:", {k: v.shape for k, v in tf_stacked_params1.items()})
print("Stacked Buffers:", tf_stacked_buffers1)

from util.torch_to_tf import torch_tensor_transpose

for k, v in tf_stacked_params1.items():
    if 'kernel' in k:
        tf_result1.append(torch_tensor_transpose(v, 1, 2))
    else:
        tf_result1.append(v)

# create multiple model instances
models_tf = [tf_SimpleNet() for _ in range(3)]

for i, network in enumerate(models_tf):
    _ = network(tf.constant(np.random.randn(1, 2).astype(np.float32)))

    if isinstance(models[i].fc, nn.Linear):
        network.fc.trainable_weights[0].assign(models[i].fc.weight.detach().numpy().T)  # kernel
        network.fc.trainable_weights[1].assign(models[i].fc.bias.detach().numpy())     # bias

# stack the parameters and buffers of these models
tf_stacked_params2, tf_stacked_buffers2 = torch_func_stack_module_state(models_tf)

print("Stacked Parameters Shape:", {k: v.shape for k, v in tf_stacked_params2.items()})
print("Stacked Buffers:", tf_stacked_buffers2)


tf_result2 = []

for k, v in tf_stacked_params2.items():
    print("type(v) = ", type(v))
    # tf_result2.append(v)
    if 'kernel' in k:
        tf_result2.append(torch_tensor_transpose(v, 1, 2))
    else:
        tf_result2.append(v)



import numpy as np

def test_stack_module_state():

    for i in range(len(result1)):
        assert np.allclose(result1[i].detach().numpy(), tf_result1[i].numpy())
        print("np.allclose(result1[i].detach().numpy(), result1[i].numpy()) = ", np.allclose(result1[i].detach().numpy(), tf_result1[i].numpy()))

    for i in range(len(result2)):
        assert np.allclose(result2[i].detach().numpy(), tf_result2[i].numpy())
        print("np.allclose(result2[i].detach().numpy(), tf_result2[i].numpy()) = ", np.allclose(result2[i].detach().numpy(), tf_result2[i].numpy()))


test_stack_module_state()































