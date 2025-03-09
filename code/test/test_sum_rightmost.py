
import tensorflow as tf

from util.torch_to_tf import _sum_rightmost

import numpy as np

# test examples
# x = tf.random.normal([2, 3, 4, 5])  # suppose this is a tensor of shape (2, 3, 4, 5)

x = tf.convert_to_tensor(np.array(range(27)))

x = tf.reshape(x, [3, 3, 3])

result = _sum_rightmost(x, 2)  # sum the last two dimension (4 and 5)

print("x = ", x)

print("result = ", result)

print(f"Input shape: {x.shape}")
print(f"Result shape: {result.shape}")
