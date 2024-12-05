import tensorflow as tf
import numpy as np

a = np.array(range(1, 10)).reshape(3, 3)

a_tensor = tf.convert_to_tensor(a)

a_shape = a.shape

zero_shape = tf.TensorShape([])

print("a_shape = ", a_shape)

print("zero_shape = ", zero_shape)

print("a_shape + zero_shape = ", a_shape + zero_shape)

