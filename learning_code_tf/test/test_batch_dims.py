import torch

# 假设 samples_expanded 的形状是 (S, B, H, A)
samples_expanded = torch.rand(3, 2, 4, 5)  # 例如 S=3, B=2, H=4, A=5
# 假设 sample_indices 的形状是 (B, 1)
sample_indices = torch.randint(0, 3, (2, 1))  # 随机选择索引，范围为 0 到 2

# 在 PyTorch 中，gather 操作如下
samples_best = torch.gather(samples_expanded, 0, sample_indices.view(1, 2, 1, 1).repeat(3, 1, 4, 5))
print(samples_best.shape)



import tensorflow as tf

# 假设 samples_expanded 的形状是 (S, B, H, A)
samples_expanded = tf.random.normal([3, 2, 4, 5])  # S=3, B=2, H=4, A=5

# 假设 sample_indices 的形状是 (B, 1)
sample_indices = tf.random.uniform([2, 1], minval=0, maxval=3, dtype=tf.int32)  # 随机选择索引，范围为 0 到 2

print("sample_indices = ", sample_indices)

# 在 TensorFlow 中，gather 操作如下
samples_best = tf.gather(samples_expanded, sample_indices, axis=0, batch_dims=1)

print("batch_dims:samples_best = ", samples_best)

print("batch_dims:samples_best.shape = ", samples_best.shape)


samples_best = tf.gather(samples_expanded, sample_indices, axis=0)

print("samples_best = ", samples_best)

print("samples_best.shape = ", samples_best.shape)


