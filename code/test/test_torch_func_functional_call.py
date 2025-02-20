


# import torch
# import torch.nn as nn
# from torch.func import functional_call

# import numpy as np


# # 定义初始化方法
# def initialize_weights_and_bias(input_dim, output_dim, seed=None):
#     if seed is not None:
#         np.random.seed(seed)  # 固定随机种子以保证一致性
#     weights = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.01
#     bias = np.random.randn(output_dim).astype(np.float32) * 0.01
#     return weights, bias

# # 设定输入和输出维度
# input_dim = 2
# output_dim = 1

# # 生成共享的初始化值
# weights, bias = initialize_weights_and_bias(input_dim, output_dim, seed=42)


# # 定义模型
# class PyTorchModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(2, 1)
        
#         # 初始化权重和偏置
#         with torch.no_grad():
#             self.fc.weight.copy_(torch.tensor(weights.T))  # PyTorch 权重是 (out_dim, in_dim)
#             self.fc.bias.copy_(torch.tensor(bias))

#     def forward(self, x):
#         return self.fc(x)

# # 初始化模型和输入
# model = PyTorchModel()
# params = dict(model.named_parameters())
# x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# y = torch.tensor([[1.0], [0.0]])

# # 计算损失
# criterion = nn.MSELoss()
# y_pred = functional_call(model, params, (x,))

# print("y_pred = ", y_pred)

# loss = criterion(y_pred, y)

# # 手动计算梯度并更新参数
# grads = torch.autograd.grad(loss, params.values())
# updated_params = {name: param - 0.1 * grad for (name, param), grad in zip(params.items(), grads)}

# # 使用 updated_params 进行新的 forward pass
# y_pred_updated = functional_call(model, updated_params, (x,))
# print("Updated Prediction:", y_pred_updated)





# import tensorflow as tf

# # 定义模型
# class TensorFlowModel(tf.keras.Model):
#     def __init__(self):
#         super(TensorFlowModel, self).__init__()
#         self.fc = tf.keras.layers.Dense(1, use_bias=True)

#         # 初始化权重和偏置
#         self.fc.build((None, input_dim))  # 必须先调用 build 方法
#         self.fc.kernel.assign(weights)   # TensorFlow 权重是 (in_dim, out_dim)
#         self.fc.bias.assign(bias)


#     def call(self, inputs):
#         return self.fc(inputs)



# # 测试 PyTorch 模型
# pytorch_model = PyTorchModel()
# print("PyTorch Weights:", pytorch_model.fc.weight)
# print("PyTorch Bias:", pytorch_model.fc.bias)

# # 测试 TensorFlow 模型
# tensorflow_model = TensorFlowModel()
# print("TensorFlow Weights:", tensorflow_model.fc.kernel.numpy())
# print("TensorFlow Bias:", tensorflow_model.fc.bias.numpy())



# # 初始化模型和输入
# model = TensorFlowModel()
# x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
# y = tf.constant([[1.0], [0.0]], dtype=tf.float32)

# # 前向传播
# with tf.GradientTape() as tape:
#     y_pred = model(x)
#     print("y_pred = ", y_pred.numpy())
#     loss = tf.reduce_mean(tf.square(y - y_pred))  # MSE Loss

# # 手动计算梯度
# params = model.trainable_variables
# grads = tape.gradient(loss, params)

# # 手动更新参数
# learning_rate = 0.1
# updated_params = [param - learning_rate * grad for param, grad in zip(params, grads)]

# # 将更新后的参数应用到模型中
# for param, updated_param in zip(params, updated_params):
#     param.assign(updated_param)

# # 使用更新后的参数进行新的前向传播
# y_pred_updated = model(x)
# print("Updated Prediction:", y_pred_updated.numpy())










import torch
import torch.nn as nn
from torch.func import functional_call

# 定义一个简单的模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.linear(x)

# 创建模型实例
model = MyModel()

# 获取模型的参数和缓冲区
params_and_buffers = {**dict(model.named_parameters()), **dict(model.named_buffers())}

print("**dict(model.named_parameters()) = ", dict(model.named_parameters()))

print("**dict(model.named_buffers()) = ", dict(model.named_buffers()))

print("params_and_buffers = ", params_and_buffers)


# 输入张量
x = torch.randn(2, 3)

# 使用 functional_call 调用模型
output = functional_call(model, params_and_buffers, (x,))
print("Output:", output)


















