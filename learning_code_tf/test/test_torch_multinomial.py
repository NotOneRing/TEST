import torch

# 权重张量
weights = torch.tensor([0.1, 0.3, 0.4, 0.2])

# 从分布中抽取 5 个样本（允许重复）
samples = torch.multinomial(weights, 5, replacement=True)
print(samples)  # 输出示例: tensor([2, 1, 3, 2, 2])


import tensorflow as tf

# 固定种子
tf.random.set_seed(42)
# logits = tf.constant([[1.0, 2.0, 3.0]])
logits = tf.constant([[0.1, 0.3, 0.4, 0.2]])

# 从该分布中采样 3 个样本
samples = tf.random.categorical(logits, num_samples=5)
print(samples)  # 输出将是可重复的


