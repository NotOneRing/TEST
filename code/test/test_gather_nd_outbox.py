import tensorflow as tf

def safe_gather_nd(tensor, indices, default_value=0.0):

    tensor_len = len(tensor.shape)
    indices_last_len = tf.shape(indices)[-1]
    augment_dim = tensor_len - indices_last_len
    

    # print("tensor = ", tensor)
    # print("indices = ", indices)
    # 获取 tensor 形状
    tensor_shape = tf.shape(tensor)
    max_indices = tensor_shape[:tf.shape(indices)[-1]]  # 计算各维度最大索引

    # print("max_indices = ", max_indices)

    # 检查索引是否越界
    is_out_of_bounds = tf.reduce_any(indices < 0, axis=-1) | tf.reduce_any(indices >= max_indices, axis=-1)

    match_dim_is_out_of_bounds = tf.expand_dims(is_out_of_bounds, axis=-1)  # (3, 1)
    
    # print("is_out_of_bounds = ", is_out_of_bounds)
    # print("match_dim_is_out_of_bounds = ", match_dim_is_out_of_bounds)
    # print("tf.zeros_like(indices) = ", tf.zeros_like(indices))
    # print("indices = ", indices)
    # 创建合法索引，将非法索引替换为 (0,0,...) 确保不会报错
    safe_indices = tf.where(match_dim_is_out_of_bounds, tf.zeros_like(indices), indices)

    # print("safe_indices = ", safe_indices)

    # 获取 gather_nd 结果
    gathered_values = tf.gather_nd(tensor, safe_indices)

    print("gathered_values = ", gathered_values)

    for i in range(augment_dim):
        is_out_of_bounds = tf.expand_dims(is_out_of_bounds, axis=-1)

    print("is_out_of_bounds = ", is_out_of_bounds)

    # 替换越界索引的部分为 default_value
    result = tf.where(is_out_of_bounds, tf.fill(tf.shape(gathered_values), tf.cast(default_value, gathered_values.dtype) ), gathered_values)

    # print("result = ", result)

    return result


# Example usage:
tensor = tf.constant([[5, 6], [7, 8]])  # shape: (2,2)
indices = tf.constant([[0, 0], [1, 1], [2, 2]])  # 最后一个索引超出范围

output = safe_gather_nd(tensor, indices)
print("output.numpy() = ", output.numpy())  # [5 8 0]

