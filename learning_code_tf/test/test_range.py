import torch

# torch.arange
tensor = torch.arange(start=0, end=10, step=2, dtype=torch.float32)
print(tensor)  # tensor([0., 2., 4., 6., 8.])


def tf_arange(start, end, step, dtype):
    return tf.range(start=start, limit=end, delta=step, dtype=dtype)

import tensorflow as tf

# tf.range
tensor = tf.range(start=0, limit=10, delta=2, dtype=tf.float32)
print(tensor)  # tf.Tensor([0. 2. 4. 6. 8.], shape=(5,), dtype=float32)


