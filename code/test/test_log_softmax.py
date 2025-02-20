import torch

x = torch.tensor([1.0, 2.0, 3.0])  # 输入张量
log_softmax_x = torch.log_softmax(x, dim=0)  # 对 dim=0 计算 log_softmax

print(log_softmax_x)
# 输出: tensor([-2.4076, -1.4076, -0.4076])





x = torch.tensor([1.0, 2.0, 3.0])
softmax = torch.exp(x) / torch.sum(torch.exp(x))  # 先计算 softmax
log_softmax_manual = torch.log(softmax)  # 再对 softmax 取对数
log_softmax_direct = torch.log_softmax(x, dim=0)  # 直接用 log_softmax

print(log_softmax_manual)
print(log_softmax_direct)  # 结果相同



x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
log_softmax_x = torch.log_softmax(x, dim=1)  # 对 dim=1 计算 log_softmax

print(log_softmax_x)



print("tensorflow!!!")





import tensorflow as tf

x = tf.constant([1.0, 2.0, 3.0])
log_softmax_x = tf.nn.log_softmax(x)

print(log_softmax_x)
# 输出: [-2.4076059 -1.4076059 -0.4076059]




x = tf.constant([1.0, 2.0, 3.0])
softmax = tf.exp(x) / tf.reduce_sum(tf.exp(x))  # 手动计算 softmax
log_softmax_manual = tf.math.log(softmax)  # 手动计算 log

log_softmax_direct = tf.nn.log_softmax(x)  # 直接用 log_softmax

print(log_softmax_manual.numpy())
print(log_softmax_direct.numpy())  # 结果相同




