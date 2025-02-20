
import tensorflow as tf


from util.torch_to_tf import torch_nn_init_trunc_normal_

tensor = tf.Variable( tf.random.uniform(shape=(30, 30), minval=10, maxval=20, dtype=tf.float32) )


print("tensor = ", tensor)

# tensor = 
torch_nn_init_trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)
# print(tensor.numpy())

print("tensor = ", tensor)

