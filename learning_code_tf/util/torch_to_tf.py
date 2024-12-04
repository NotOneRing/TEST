import tensorflow as tf
import numpy as np

def tf_index_gather(input_tensor, dim, index_tensor):
    """
    Mimics the behavior of indexing in PyTorch. 
    Specifically:
        - out[i][j][k] = input[index[i][j][k]][j][k]  if dim == 0
        - out[i][j][k] = input[i][index[i][j][k]][k]  if dim == 1
        - out[i][j][k] = input[i][j][index[i][j][k]]  if dim == 2
    
    Args:
        input_tensor (tf.Tensor): The input tensor from which to gather values.
        index_tensor (tf.Tensor): The indices tensor.
        dim (int): The dimension along which to gather the values.
    
    Returns:
        tf.Tensor: The output tensor with the gathered values.
    """

    assert input_tensor.shape.as_list() == index_tensor.shape.as_list(), "input_tensor.shape is not equal to index_tensor.shape"

    index_array = index_tensor.numpy()

    input_array = input_tensor.numpy()

    dim_list = input_tensor.shape.as_list()

    dim_number = len(input_tensor.shape)

    cur_index = [0] * dim_number
    
    output_matrix = np.zeros(dim_list, dtype=np.int64)

    import math
    total_number = math.prod(dim_list)

    count_number = 0

    while count_number < total_number:

        for cur_dim_index in range(dim_list[dim_number - 1]):
            cur_index[dim_number-1] = cur_dim_index
            dim_true_index = index_array[ tuple(cur_index) ]
            dim_ori_index = cur_index[dim]
            cur_index[dim] = dim_true_index
            val = input_array[ tuple(cur_index) ]
            cur_index[dim] = dim_ori_index
            output_matrix[tuple(cur_index)] = val
            count_number += 1

        cur_index[dim_number - 1] += 1

        for i in range(dim_number - 1, 0, -1):
            if cur_index[i] > dim_list[i] - 1:
                cur_index[i] -= (dim_list[i])
                cur_index[i-1] += 1
            else:
                break

        if cur_index[0] > dim_list[0] - 1:
            break

    output_matrix = tf.convert_to_tensor(output_matrix)
    
    return output_matrix




def tf_quantile(input_tensor, q, dim=None, interpolation='linear'):
    """
    Compute the quantile of the input_tensor along a specified axis using TensorFlow.
    """

    # Flatten the input if axis is None
    if dim is None:
        input_tensor = tf.reshape(input_tensor, [-1])
        dim = 0

    # Sort the tensor along the given axis
    sorted_tensor = tf.sort(input_tensor, axis=dim)

    # Get the size of the specified axis
    n = tf.cast(tf.shape(sorted_tensor)[dim], tf.float32)
    
    # Compute the indices for quantiles
    indices = q * (n - 1)

    lower_indices = tf.math.floor(indices)
    upper_indices = tf.math.ceil(indices)

    # Gather values at lower and upper indices
    lower_values = tf.gather(sorted_tensor, tf.cast(lower_indices, tf.int32), axis=dim)
    upper_values = tf.gather(sorted_tensor, tf.cast(upper_indices, tf.int32), axis=dim)
    
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





def tf_flatten(input_tensor, start_dim = 0, end_dim = -1):
    tensor_shape_list = input_tensor.shape.as_list()

    if end_dim == -1:
        end_dim = len(tensor_shape_list) - 1

    middle_dim = 1
    for i in range(start_dim, end_dim + 1):
        middle_dim *= tensor_shape_list[i]
    shape_list = []
    for i in range(0, start_dim):
        shape_list.append(tensor_shape_list[i])
    shape_list.append(middle_dim)
    for i in range(end_dim + 1, len(tensor_shape_list)):
        shape_list.append(tensor_shape_list[i])

    output_tensor = tf.reshape(input_tensor, shape_list)
    return output_tensor






def tf_arange(start, end, step, dtype):
    return tf.range(start=start, limit=end, delta=step, dtype=dtype)







class Normal:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


    def log_prob(self, x):
        """
        计算正态分布的对数概率密度函数

        Args:
            x: 需要计算概率密度的点
            mean: 正态分布的均值
            std: 正态分布的标准差

        Returns:
            对数概率密度
        """
        log_pdf = -tf.math.log(self.std * tf.math.sqrt(2 * tf.constant(np.pi))) - 0.5 * ((x - self.mean) ** 2) / (self.std ** 2)
        return log_pdf


# class Categorical:
#     # >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
#     # >>> m.sample()  # equal probability of 0, 1, 2, 3
#     # tensor(3)

#     def __init__(self, logits):
#         self.logits = logits

#     def log_prob(self, x):
#         pass


class Categorical:
    def __init__(self, probs=None, logits=None):
        
        print("logits.shape = ", logits.shape)

        if probs is not None:
            self.probs = probs
        elif logits is not None:
            self.logits = logits
            self.probs = tf.nn.softmax(logits, axis=-1)
        else:
            raise ValueError("Must specify either probs or logits.")
    
    def sample(self):
        return tf.random.categorical(self.probs, num_samples = 1, dtype=tf.int32)

    def log_prob(self, value):
        assert len(value.shape.as_list()) <= 2
        if self.probs is not None:

            value_shape_list = list(value.shape)

            batch_dim = value_shape_list[0]

            all_tensors = []

            for i in range(batch_dim):
                index = int(value[i, ...].numpy())  # 获取索引

                log_prob_value = tf.gather(self.probs, index, axis=-1)  # 从 probs 中收集数据

                # 然后计算 log
                log_prob_value = tf.math.log(log_prob_value)

                # 将结果重新形状化
                all_tensors.append(tf.reshape(log_prob_value, [1, -1]))
                

            if batch_dim == 1:
                result = all_tensors[0]
            else:
                result = tf.concat(all_tensors, axis=0)

            return result

        else:  # logits provided
            raise ValueError("Must specify probs.")


    def entropy(self):
        return -tf.reduce_sum( self.probs * tf.math.log(self.probs), axis=-1 )













class Independent:
    def __init__(self):
        pass

    def log_prob(self, x):
        pass















class MixtureSameFamily:
    def __init__(self):
        pass

    def log_prob(self, x):
        pass









def F_mse_loss(input_tensor1, input_tensor2):
    return tf.keras.losses.MeanSquaredError(input_tensor1, input_tensor2)



def torch_min(input, dim = None, other = None):
    if other == None:
        return tf.reduce_min(input, axis=dim)
    else:
        return tf.minimum(input, other)

def torch_max(input, dim = None, other = None):
    if other == None:
        return tf.reduce_min(input, axis=dim)
    else:
        return tf.maximum(input, other)


def torch_mean(input_tensor1, dim = None, keepdim = False):
    return tf.reduce_mean(input_tensor1, axis = dim, keepdims = keepdim)



def torch_softmax(input_tensor, dim):
    return tf.nn.softmax(input_tensor, axis=dim)


def torch_stack(tensor_list_to_stack, dim):
    return tf.stack(tensor_list_to_stack, axis = dim)


def torch_multinomial(input, num_samples, replacement = True):
    assert replacement == True, "replacement must be True to use tf.random.categorical"
    return tf.random.categorical(input, num_samples=num_samples)




def torch_where(index_tensor, input_tensor, replace_value):
    return tf.where(index_tensor, input_tensor, replace_value)




def torch_vmap():
    pass



def torch_func_stack_module_state():
    pass




def torch_tensor(input_numpy_array):
    return tf.convert_to_tensor(input_numpy_array)




def torch_clamp(input, min = float('-inf'), max = float('inf'), out=None):
    out = tf.clip_by_value(input, min, max)
    return out
