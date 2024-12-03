
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



def tf_quantile(input_tensor, q, dim=None, interpolation='linear'):
    """
    Compute the quantile of the input_tensor along a specified axis using TensorFlow.

    Args:
        input_tensor (tf.Tensor): Input tensor.
        q (float or list): Quantile value(s) in [0, 1]. Can be a single float or a list of floats.
        axis (int, optional): The axis along which to compute the quantile. Default is None (flatten the tensor).
        interpolation (str, optional): Interpolation method. Options are 'linear', 'lower', 'higher', 'nearest', 'midpoint'.

    Returns:
        tf.Tensor: Tensor of quantile values.
    """
    # Ensure q is a tensor
    # q = tf.convert_to_tensor(q, dtype=tf.float32)
    
    # Flatten the input if axis is None
    if dim is None:
        input_tensor = tf.reshape(input_tensor, [-1])
        dim = 0

    # Sort the tensor along the given axis
    sorted_tensor = tf.sort(input_tensor, axis=dim)

    print("sorted_tensor = ", sorted_tensor)
    
    # Get the size of the specified axis
    n = tf.cast(tf.shape(sorted_tensor)[dim], tf.float32)
    
    # Compute the indices for quantiles
    indices = q * (n - 1)
    lower_indices = tf.math.floor(indices)
    upper_indices = tf.math.ceil(indices)

    print("lower_indices = ", lower_indices)
    print("upper_indices = ", upper_indices)

    # Gather values at lower and upper indices
    lower_values = tf.gather(sorted_tensor, tf.cast(lower_indices, tf.int32), axis=dim)
    upper_values = tf.gather(sorted_tensor, tf.cast(upper_indices, tf.int32), axis=dim)

    print("lower_values = ", lower_values)
    print("upper_values = ", upper_values)
    
    # Compute weights for linear interpolation
    weights = indices - lower_indices

    # Interpolation methods
    if interpolation == 'linear':
        result = (1 - weights) * lower_values + weights * upper_values
    elif interpolation == 'lower':
        result = lower_values
    elif interpolation == 'higher':
        result = upper_values
    elif interpolation == 'nearest':
        result = tf.where(weights < 0.5, lower_values, upper_values)
    elif interpolation == 'midpoint':
        result = (lower_values + upper_values) / 2
    else:
        raise ValueError("Unsupported interpolation method: {}".format(interpolation))
    
    return result

a = tf.convert_to_tensor(a.numpy())

q = tf.convert_to_tensor(q.numpy())


temp = tf_quantile(a, q, dim=1)

print("temp = ", temp)
print("temp.shape = ", temp.shape)














