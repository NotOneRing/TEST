import tensorflow as tf

# 定义一个示例张量
# tensor = tf.constant([[1, 2, 3], 
#                       [4, 5, 6], 
#                       [7, 8, 9]])
tensor = tf.range(0, 27, 1)

tensor = tf.reshape(tensor, (3, 3, 3))

print("tensor = ", tensor)

# 定义索引列表
indices = tf.constant([[0, 0], [1, 1], [2, 2]])

# 使用 tf.gather_nd
result = tf.gather_nd(tensor, indices)

print(result.numpy())  # 输出 [1, 5, 9]



# 假设需要沿第 0 维和第 1 维同时索引
rows = tf.constant([0, 1, 2])  # 第 0 维索引
cols = tf.constant([0, 1, 2])  # 第 1 维索引

# 使用 tf.stack 创建索引
indices = tf.stack([rows, cols], axis=1)

# 使用 tf.gather_nd 提取值
result = tf.gather_nd(tensor, indices)

print(result.numpy())  # 输出 [1, 5, 9]


