


# import torch
# import torch.nn as nn
# from torch.func import functional_call

# import numpy as np


# # define initialization
# def initialize_weights_and_bias(input_dim, output_dim, seed=None):
#     if seed is not None:
#         np.random.seed(seed)  # fix random seeds to ensure reproducibility
#     weights = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.01
#     bias = np.random.randn(output_dim).astype(np.float32) * 0.01
#     return weights, bias

# # set input and output dimension
# input_dim = 2
# output_dim = 1

# # generate shared initialization values
# weights, bias = initialize_weights_and_bias(input_dim, output_dim, seed=42)


# # define the model
# class PyTorchModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(2, 1)
        
#         # initialize weights and biases
#         with torch.no_grad():
#             self.fc.weight.copy_(torch.tensor(weights.T))  # PyTorch's weight is of (out_dim, in_dim)
#             self.fc.bias.copy_(torch.tensor(bias))

#     def forward(self, x):
#         return self.fc(x)

# # initialize the model and the input
# model = PyTorchModel()
# params = dict(model.named_parameters())
# x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# y = torch.tensor([[1.0], [0.0]])

# # calculate loss
# criterion = nn.MSELoss()
# y_pred = functional_call(model, params, (x,))

# print("y_pred = ", y_pred)

# loss = criterion(y_pred, y)

# # calculate grad and update parameters manually
# grads = torch.autograd.grad(loss, params.values())
# updated_params = {name: param - 0.1 * grad for (name, param), grad in zip(params.items(), grads)}

# # use updated_params to have new forward pass
# y_pred_updated = functional_call(model, updated_params, (x,))
# print("Updated Prediction:", y_pred_updated)





# import tensorflow as tf

# # define model
# class TensorFlowModel(tf.keras.Model):
#     def __init__(self):
#         super(TensorFlowModel, self).__init__()
#         self.fc = tf.keras.layers.Dense(1, use_bias=True)

#         # initialize weights and biases
#         self.fc.build((None, input_dim))  # first call build method
#         self.fc.kernel.assign(weights)   # TensorFlow's weight is of (in_dim, out_dim)
#         self.fc.bias.assign(bias)


#     def call(self, inputs):
#         return self.fc(inputs)



# # test PyTorch model
# pytorch_model = PyTorchModel()
# print("PyTorch Weights:", pytorch_model.fc.weight)
# print("PyTorch Bias:", pytorch_model.fc.bias)

# # test TensorFlow model
# tensorflow_model = TensorFlowModel()
# print("TensorFlow Weights:", tensorflow_model.fc.kernel.numpy())
# print("TensorFlow Bias:", tensorflow_model.fc.bias.numpy())



# # initialize model and input
# model = TensorFlowModel()
# x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
# y = tf.constant([[1.0], [0.0]], dtype=tf.float32)

# # forward pass
# with tf.GradientTape() as tape:
#     y_pred = model(x)
#     print("y_pred = ", y_pred.numpy())
#     loss = tf.reduce_mean(tf.square(y - y_pred))  # MSE Loss

# # calculate gradient manually
# params = model.trainable_variables
# grads = tape.gradient(loss, params)

# # update parameters manually
# learning_rate = 0.1
# updated_params = [param - learning_rate * grad for param, grad in zip(params, grads)]

# # apply updated parameters to the model
# for param, updated_param in zip(params, updated_params):
#     param.assign(updated_param)

# # use updated parameters to have the forward pass
# y_pred_updated = model(x)
# print("Updated Prediction:", y_pred_updated.numpy())










import torch
import torch.nn as nn
from torch.func import functional_call

# define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.linear(x)

# create a model instance
model = MyModel()

# get model's parameter and buffer
params_and_buffers = {**dict(model.named_parameters()), **dict(model.named_buffers())}

print("**dict(model.named_parameters()) = ", dict(model.named_parameters()))

print("**dict(model.named_buffers()) = ", dict(model.named_buffers()))

print("params_and_buffers = ", params_and_buffers)


# input tensor
x = torch.randn(2, 3)

# use functional_call to call model
output = functional_call(model, params_and_buffers, (x,))
print("Output:", output)


















