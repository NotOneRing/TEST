import tensorflow as tf

from util.torch_to_tf import torch_nn_utils_clip_grad_norm_

# def torch_nn_utils_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):


# # 假设我们有一个模型
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
#     tf.keras.layers.Dense(1)
# ])

# # 获取模型参数
# parameters = model.trainable_variables

# # 创建一个优化器
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# # 创建输入数据
# x = tf.random.normal([32, 5])
# y = tf.random.normal([32, 1])

# # 使用梯度带计算梯度
# with tf.GradientTape() as tape:
#     output = model(x)
#     loss = tf.reduce_mean(tf.square(output - y))

# # 计算梯度
# grads = tape.gradient(loss, parameters)

# # 裁剪梯度
# clipped_grads = tensorflow_clip_grad_norm_(grads, max_norm=1.0)

# # 应用裁剪后的梯度
# optimizer.apply_gradients(zip(clipped_grads, parameters))

# print("Training finished.")







import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tensorflow as tf


from util.torch_to_tf import torch_optim_Adam, nn_Linear, nn_ReLU, nn_Sequential



# 设置随机种子，确保可复现性
torch.manual_seed(0)
np.random.seed(0)
tf.random.set_seed(0)



# TensorFlow 模型
tf_model = nn_Sequential([
    nn_Linear(5, 10),
    nn_ReLU(),
    nn_Linear(10, 1)
])

# PyTorch 模型
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.linear1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

torch_model = PyTorchModel()


tf_model.build( input_shape = (None, 5) )

# #后加的，为了初始化模型
_ = tf_model(tf.constant(np.random.randn(1, 5).astype(np.float32)))


# 同步初始化权重 (尽量接近)
with torch.no_grad():
    # print("dir(tf_model[0]) = ", dir(tf_model[0]))
    print("tf_model = ", tf_model)
    print("tf_model[0] = ", tf_model[0])
    print("tf_model[0].trainable_variables = ", tf_model[0].trainable_variables)
    torch_model.linear1.weight.copy_(torch.from_numpy(tf_model[0].model.kernel.numpy().T))
    torch_model.linear1.bias.copy_(torch.from_numpy(tf_model[0].model.bias.numpy()))
    torch_model.linear2.weight.copy_(torch.from_numpy(tf_model[2].model.kernel.numpy().T))
    torch_model.linear2.bias.copy_(torch.from_numpy(tf_model[2].model.bias.numpy()))



# 定义优化器
tf_optimizer = torch_optim_Adam(tf_model.trainable_variables, lr=0.01)
torch_optimizer = optim.Adam(torch_model.parameters(), lr=0.01)


# 创建输入数据 (保持一致)
tf_x = tf.random.normal([32, 5])
tf_y = tf.random.normal([32, 1])
torch_x = torch.tensor(tf_x.numpy(), dtype=torch.float32)
torch_y = torch.tensor(tf_y.numpy(), dtype=torch.float32)


# TensorFlow 训练和梯度裁剪
with tf.GradientTape() as tape:
    tf_output = tf_model(tf_x)
    tf_loss = tf.reduce_mean(tf.square(tf_output - tf_y))
tf_grads = tape.gradient(tf_loss, tf_model.trainable_variables)

clipped_tf_grads = torch_nn_utils_clip_grad_norm_(tf_model.trainable_variables, max_norm=1.0, grads = tf_grads)

# 计算 TensorFlow 裁剪后梯度的范数
clipped_tf_grad_norm = tf.norm(tf.stack([tf.norm(g) for g in clipped_tf_grads if g is not None]))
print(f"TensorFlow clipped grad norm: {clipped_tf_grad_norm.numpy()}")

# temp = zip(clipped_tf_grads, tf_model.trainable_variables)

# print("temp = ", temp)
# print( "type(temp) = ", type(temp) )

# tf_optimizer.apply_gradients(zip(clipped_tf_grads, tf_model.trainable_variables))
tf_optimizer.apply_gradients(clipped_tf_grads)

print("tensorflow Training finished.")







# PyTorch 训练和梯度裁剪
torch_output = torch_model(torch_x)
torch_loss = torch.nn.functional.mse_loss(torch_output, torch_y)
torch_optimizer.zero_grad()
torch_loss.backward()
torch.nn.utils.clip_grad_norm_(torch_model.parameters(), max_norm=1.0)

# 计算 PyTorch 裁剪后梯度的范数
total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in torch_model.parameters() if p.grad is not None]))
print(f"PyTorch clipped grad norm: {total_norm.item()}")


torch_optimizer.step()
print("pytorch Training finished.")





























