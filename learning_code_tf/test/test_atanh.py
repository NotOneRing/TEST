import tensorflow as tf
import torch
import numpy as np

from util.torch_to_tf import torch_atanh

x = tf.constant([0.5, 0.1, 0.9, 1.5, -0.5, -1, -3], dtype=tf.float32)
y = torch_atanh(x)

print(y)

x_torch = torch.tensor(x.numpy())

y_torch = torch.atanh(x_torch)

print(y_torch)




# 检查是否为 NaN 或 Inf
y_is_nan = np.isnan(y.numpy())
y_torch_is_nan = np.isnan(y_torch.numpy())

y_is_inf = np.isinf(y.numpy())
y_torch_is_inf = np.isinf(y_torch.numpy())



# 先检查 NaN 和 Inf 是否一致
if np.all(y_is_nan == y_torch_is_nan) and np.all(y_is_inf == y_torch_is_inf):
    # 忽略 NaN 和 Inf，检查其它值
    result = np.allclose(y.numpy()[~(y_is_nan | y_is_inf)], y_torch.numpy()[~(y_torch_is_nan | y_torch_is_inf)], atol=1e-4)
else:
    result = False

print(result)


