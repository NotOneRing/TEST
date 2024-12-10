import torch

# torch.dot                            # [D], [D] -> []
batched_dot = torch.vmap(torch.vmap(torch.dot))  # [N1, N0, D], [N1, N0, D] -> [N1, N0]
x, y = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
print("x = ", x)
print("y = ", y)
result = batched_dot(x, y) # tensor of size [2, 3]

print("result = ", result)



# torch.dot                            # [N], [N] -> []
batched_dot = torch.vmap(torch.dot, in_dims=1)  # [N, D], [N, D] -> [D]
x, y = torch.randn(2, 5), torch.randn(2, 5)
print("x = ", x)
print("y = ", y)
result2 = batched_dot(x, y)   # output is [5] instead of [2] if batched along the 0th dimension

print("result2 = ", result2)






import tensorflow as tf

# 定义一个简单的标量函数
def scalar_function(x):
    return tf.reduce_sum(x ** 2)

# 输入数据：batch_size=5，每个样本有3个特征
inputs = tf.random.normal([5, 3])


print("inputs = ", inputs)

print("inputs**2 = ", inputs**2)

# 使用 tf.vectorized_map 对每个样本应用函数
outputs = tf.vectorized_map(scalar_function, inputs)

print("Outputs:", outputs)
















