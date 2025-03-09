import torch
import tensorflow as tf
import numpy as np

# # set random seed to make sure the reproducibility
# np.random.seed(42)
# torch.manual_seed(42)
# tf.random.set_seed(42)

# create a simple probability distribution with 3 categories
probabilities = np.array([0.2, 0.5, 0.3])  # probability of each category

# number of samples
num_samples = 5

# ---------------------------------
# 1. use torch.multinomial to sample
# convert to torch tensor
torch_probs = torch.tensor(probabilities, dtype=torch.float32)

# torch.multinomial sample
torch_samples = torch.multinomial(torch_probs, num_samples, replacement=True)
print(f"torch.multinomial sampled indices: {torch_samples.numpy()}")

# ---------------------------------
# 2. use tensorflow's tf.random.categorical to sample
# convert to TensorFlow tensor
tf_probs = tf.convert_to_tensor(probabilities, dtype=tf.float32)

# use tf.random.categorical to sample
# tf.random.categorical use logits, so we convert tf_probs into logits
logits = tf.math.log(tf_probs)

# num_samples is the number of samples to draw
tf_samples = tf.random.categorical(logits[None, :], num_samples, dtype=tf.int32).numpy().flatten()

print(f"tensorflow tf.random.categorical sampled indices: {tf_samples}")
