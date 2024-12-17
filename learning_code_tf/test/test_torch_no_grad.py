
import tensorflow as tf

x = tf.Variable([1.0, 2.0, 3.0], trainable=True)

# with torch.no_grad():

with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
    y = x * 2
    z = y + 1  # y 的梯度计算被禁止
    loss = tf.reduce_sum(z)

# 梯度计算
grad = tape.gradient(loss, x)
print("x grad = ", grad)  # 输出 None，因为 z 不会传播梯度

grad = tape.gradient(loss, y)
print("y grad = ", grad)


grad = tape.gradient(loss, z)
print("z grad = ", grad)

from util.torch_to_tf import torch_no_grad


with torch_no_grad() as tape:
    y = x * 2
    z = y + 1  # y 的梯度计算被禁止
    loss = tf.reduce_sum(z)

# 梯度计算
grad = tape.gradient(loss, x)
print("x grad = ", grad)  # 输出 None，因为 z 不会传播梯度

grad = tape.gradient(loss, y)
print("y grad = ", grad)


grad = tape.gradient(loss, z)
print("z grad = ", grad)



