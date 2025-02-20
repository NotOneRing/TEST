import torch
import tensorflow as tf
import numpy as np

# # 设置随机种子以确保结果可复现
# np.random.seed(42)
# torch.manual_seed(42)
# tf.random.set_seed(42)

# 生成一个简单的概率分布，3个类别
probabilities = np.array([0.2, 0.5, 0.3])  # 类别的概率分布

# 样本数量
num_samples = 5

# ---------------------------------
# 1. 使用 torch.multinomial 采样
# 转换为 torch tensor
torch_probs = torch.tensor(probabilities, dtype=torch.float32)

# torch.multinomial 采样
torch_samples = torch.multinomial(torch_probs, num_samples, replacement=True)
print(f"torch.multinomial sampled indices: {torch_samples.numpy()}")

# ---------------------------------
# 2. 使用 tensorflow 中的 tf.random.categorical 采样
# 转换为 TensorFlow tensor
tf_probs = tf.convert_to_tensor(probabilities, dtype=tf.float32)

# tf.random.categorical 采样
# tf.random.categorical 接受 logits，因此需要将概率转换为 logits
logits = tf.math.log(tf_probs)

# 采样，num_samples 为采样数
tf_samples = tf.random.categorical(logits[None, :], num_samples, dtype=tf.int32).numpy().flatten()

print(f"tensorflow tf.random.categorical sampled indices: {tf_samples}")
