import tensorflow as tf





import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tensorflow as tf


from util.torch_to_tf import torch_optim_Adam, nn_Linear, nn_ReLU, nn_Sequential, model_forward_backward_gradients, torch_nn_utils_clip_grad_norm_and_step



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

#initialize model
_ = tf_model(tf.constant(np.random.randn(1, 5).astype(np.float32)))


# 同步初始化权重 (尽量接近)
with torch.no_grad():
    # print("tf_model = ", tf_model)
    # print("tf_model[0] = ", tf_model[0])
    # print("tf_model[0].trainable_variables = ", tf_model[0].trainable_variables)
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


# # TensorFlow 训练和梯度裁剪
# with tf.GradientTape() as tape:
#     tf_output = tf_model(tf_x)
#     tf_loss = tf.reduce_mean(tf.square(tf_output - tf_y))
# tf_grads = tape.gradient(tf_loss, tf_model.trainable_variables)

# clipped_tf_grads = torch_nn_utils_clip_grad_norm_and_step(tf_model.trainable_variables, tf_optimizer, max_norm=1.0, grads = tf_grads)

# stacked_tf_clipped_grad = tf.stack([tf.norm(g) for g in clipped_tf_grads if g is not None])
# # 计算 TensorFlow 裁剪后梯度的范数
# clipped_tf_grad_norm = tf.norm(stacked_tf_clipped_grad)

# print("stacked_tf_clipped_grad = ", stacked_tf_clipped_grad)

# print(f"TensorFlow clipped grad norm: {clipped_tf_grad_norm.numpy()}")

# # temp = zip(clipped_tf_grads, tf_model.trainable_variables)

# # print("temp = ", temp)
# # print( "type(temp) = ", type(temp) )

# # tf_optimizer.apply_gradients(zip(clipped_tf_grads, tf_model.trainable_variables))
# # tf_optimizer.apply_gradients(clipped_tf_grads)

# print("tensorflow Training finished.")







# # PyTorch 训练和梯度裁剪
# torch_output = torch_model(torch_x)
# torch_loss = torch.nn.functional.mse_loss(torch_output, torch_y)
# torch_optimizer.zero_grad()
# torch_loss.backward()
# torch.nn.utils.clip_grad_norm_(torch_model.parameters(), max_norm=1.0)

# clipped_grad = torch.stack([torch.norm(p.grad) for p in torch_model.parameters() if p.grad is not None])

# print("PyTorch: clipped_grad = ", clipped_grad)

# # 计算 PyTorch 裁剪后梯度的范数
# total_norm = torch.norm(clipped_grad)
# print(f"PyTorch clipped grad norm: {total_norm.item()}")


# torch_optimizer.step()
# print("pytorch Training finished.")


# print("np.allclose( stacked_tf_clipped_grad.numpy(), clipped_grad.numpy() ) = ", np.allclose( stacked_tf_clipped_grad.numpy(), clipped_grad.numpy() ))


# assert np.allclose( stacked_tf_clipped_grad.numpy(), clipped_grad.numpy() )







# Training loop
for step in range(5):
    
    # PyTorch 训练和梯度裁剪
    torch_optimizer.zero_grad()
    torch_output = torch_model(torch_x)
    torch_loss = torch.nn.functional.mse_loss(torch_output, torch_y)
    torch_loss.backward()
    torch.nn.utils.clip_grad_norm_(torch_model.parameters(), max_norm=1.0)

    clipped_grad = torch.stack([torch.norm(p.grad) for p in torch_model.parameters() if p.grad is not None])

    print("PyTorch: clipped_grad = ", clipped_grad)

    # 计算 PyTorch 裁剪后梯度的范数
    total_norm = torch.norm(clipped_grad)
    print(f"PyTorch clipped grad norm: {total_norm.item()}")

    torch_optimizer.step()
    print("pytorch Training finished.")



    tf_loss_fn = lambda x, y: tf.reduce_mean(tf.square(x - y))

    tf_loss, tf_gradients = model_forward_backward_gradients(tf_x, tf_y, tf_loss_fn, tf_model)


    # with tf.GradientTape() as tape:
    #     tf_output = tf_model(tf_x)
    #     tf_loss = tf.reduce_mean(tf.square(tf_output - tf_y))
    # tf_grads = tape.gradient(tf_loss, tf_model.trainable_variables)

    clipped_tf_grads = torch_nn_utils_clip_grad_norm_and_step(tf_model.trainable_variables, tf_optimizer, max_norm=1.0, grads = tf_gradients)

    stacked_tf_clipped_grad = tf.stack([tf.norm(g) for g in clipped_tf_grads if g is not None])
    # 计算 TensorFlow 裁剪后梯度的范数
    clipped_tf_grad_norm = tf.norm(stacked_tf_clipped_grad)

    print("stacked_tf_clipped_grad = ", stacked_tf_clipped_grad)

    print(f"TensorFlow clipped grad norm: {clipped_tf_grad_norm.numpy()}")


    print("np.allclose( stacked_tf_clipped_grad.numpy(), clipped_grad.numpy() ) = ", np.allclose( stacked_tf_clipped_grad.numpy(), clipped_grad.numpy() ))

    # # TensorFlow
    # with tf.GradientTape() as tape:
    #     tf_outputs = tf_model(inputs)
    #     tf_loss = tf_loss_fn(targets, tf_outputs)
    # tf_gradients = tape.gradient(tf_loss, tf_model.trainable_variables)

    # tf_optimizer.step(tf_gradients)

    # Print losses
    print(f"Step {step + 1}:")
    print(f"  PyTorch Loss: {torch_loss.item():.6f}")
    print(f"  TensorFlow Loss: {tf_loss.numpy():.6f}")

    # print("np.allclose(torch_loss.item(), tf_loss.numpy()) = ", np.allclose(torch_loss.item(), tf_loss.numpy()) )
    assert np.allclose(torch_loss.item(), tf_loss.numpy(), atol = 1e-6)

# Compare final outputs
torch_final_output = torch_model(torch.tensor(torch_x)).detach().numpy()
tf_final_output = tf_model(tf_x).numpy()

print("\nFinal Output Comparison:")
print(f"  PyTorch: {torch_final_output}")
print(f"  TensorFlow: {tf_final_output}")
print("np.allclose(torch_final_output, tf_final_output) = ", np.allclose(torch_final_output, tf_final_output, atol=1e-4) )
assert np.allclose(torch_final_output, tf_final_output, atol=1e-4)



















