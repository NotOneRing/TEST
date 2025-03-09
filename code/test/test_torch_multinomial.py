import torch

# weights tensor
weights = torch.tensor([0.1, 0.3, 0.4, 0.2])

# take out 5 samples from the distribution(sampling with replacement)
samples = torch.multinomial(weights, 5, replacement=True)
print(samples)  # output examples: tensor([2, 1, 3, 2, 2])


import tensorflow as tf

# fix seed
tf.random.set_seed(42)
# logits = tf.constant([[1.0, 2.0, 3.0]])
logits = tf.constant([[0.1, 0.3, 0.4, 0.2]])

# take out 3 samples from the distribution(sampling with replacement)
samples = tf.random.categorical(logits, num_samples=5)
print(samples)  # outputs could repeat














