
import tensorflow as tf

from util.torch_to_tf import _sum_rightmost

import numpy as np

# 测试例子
# x = tf.random.normal([2, 3, 4, 5])  # 假设这是一个形状为 (2, 3, 4, 5) 的张量

x = tf.convert_to_tensor(np.array(range(27)))

x = tf.reshape(x, [3, 3, 3])

result = _sum_rightmost(x, 2)  # 对最后两个维度 (4 和 5) 求和

print("x = ", x)

print("result = ", result)

print(f"Input shape: {x.shape}")
print(f"Result shape: {result.shape}")
