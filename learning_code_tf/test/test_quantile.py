
import torch

import tensorflow as tf

import numpy as np



# advantages = np.array([1, 2, 3, 4, 5], dtype=np.float32)

# advantages = torch.tensor(advantages)

# print("advantages = ", advantages)

# self_clip_advantage_lower_quantile = 0.1
# self_clip_advantage_upper_quantile = 0.9

# # Clip advantages by 5th and 95th percentile
# advantage_min = torch.quantile(advantages, self_clip_advantage_lower_quantile)

# print("advantage_min = ", advantage_min)

# advantage_max = torch.quantile(advantages, self_clip_advantage_upper_quantile)

# print("advantage_max = ", advantage_max)

# percent = 0.25

# advantage = torch.quantile(advantages, percent)

# print("advantage = ", advantage)


# advantages = advantages.clamp(min=advantage_min, max=advantage_max)

# print("advantages = ", advantage_max)

# print("advantages = ", advantages)

# advantages = tf.convert_to_tensor(advantages)

# advantage_min = tfp.stats.percentile(advantages, self_clip_advantage_lower_quantile * 100)
# advantage_max = tfp.stats.percentile(advantages, self_clip_advantage_upper_quantile * 100)
# advantages = tf.clip_by_value(advantages, advantage_min, advantage_max)

# print("advantages = ", advantages)




a = torch.tensor([[ 0.0795, -1.2117,  0.9765],
        [ 1.1707,  0.6706,  0.4884]])

print("a.shape = ", a.shape)

q = torch.tensor([0.25, 0.5, 0.75])

temp = torch.quantile(a, q, dim=1, keepdim=True)

print("temp = ", temp)
print("temp.shape = ", temp.shape)

# tensor([[[-0.5661],
#         [ 0.5795]],
#         [[ 0.0795],
#         [ 0.6706]],
#         [[ 0.5280],
#         [ 0.9206]]])



a = tf.convert_to_tensor(a.numpy())

q = tf.convert_to_tensor(q.numpy())


temp = tf_quantile(a, q, dim=1)

print("temp = ", temp)
print("temp.shape = ", temp.shape)














