
import tensorflow as tf

x = tf.Variable([1.0, 2.0, 3.0], trainable=True)

# with torch.no_grad():

with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
    y = x * 2
    z = y + 1
    loss = tf.reduce_sum(z)

# calculate gradient
grad = tape.gradient(loss, x)
print("x grad = ", grad)  # output grad

grad = tape.gradient(loss, y)
print("y grad = ", grad)


grad = tape.gradient(loss, z)
print("z grad = ", grad)

from util.torch_to_tf import torch_no_grad


with torch_no_grad() as tape:
    y = x * 2
    z = y + 1  # y's gradient calculation is forbidden by torch_no_grad
    loss = tf.reduce_sum(z)

# calculate gradient
grad = tape.gradient(loss, x)
print("x grad = ", grad)  # Output None, because z does not propagate gradient

grad = tape.gradient(loss, y)
print("y grad = ", grad)


grad = tape.gradient(loss, z)
print("z grad = ", grad)



