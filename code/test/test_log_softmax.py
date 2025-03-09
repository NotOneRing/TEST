import torch

x = torch.tensor([1.0, 2.0, 3.0])  # input tensor
log_softmax_x = torch.log_softmax(x, dim=0)  # calculate log_softmax for dim=0

print(log_softmax_x)
# output: tensor([-2.4076, -1.4076, -0.4076])





x = torch.tensor([1.0, 2.0, 3.0])
softmax = torch.exp(x) / torch.sum(torch.exp(x))  # calculate softmax
log_softmax_manual = torch.log(softmax)  # get log for softmax
log_softmax_direct = torch.log_softmax(x, dim=0)  # use log_softmax directly

print(log_softmax_manual)
print(log_softmax_direct)  # equivalent results



x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
log_softmax_x = torch.log_softmax(x, dim=1)  # use dim=1 to calculate log_softmax

print(log_softmax_x)



print("tensorflow!!!")





import tensorflow as tf

x = tf.constant([1.0, 2.0, 3.0])
log_softmax_x = tf.nn.log_softmax(x)

print(log_softmax_x)
# output: [-2.4076059 -1.4076059 -0.4076059]




x = tf.constant([1.0, 2.0, 3.0])
softmax = tf.exp(x) / tf.reduce_sum(tf.exp(x))  # calculate softmax manually
log_softmax_manual = tf.math.log(softmax)  # calculate log manually

log_softmax_direct = tf.nn.log_softmax(x)  # use log_softmax directly

print(log_softmax_manual.numpy())
print(log_softmax_direct.numpy())  # the same result




