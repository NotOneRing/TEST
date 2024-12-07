import tensorflow as tf
import numpy as np


def torch_gather(input_tensor, dim, index_tensor):
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





def torch_flatten(input_tensor, start_dim = 0, end_dim = -1):
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




# import torch


class Normal:
    def __init__(self, loc, scale):
        #mean
        self.loc = loc

        # print("self.loc = ", self.loc)

        #std
        self.scale = scale

        self.batch_shape = self.loc.shape

        self.event_shape = tf.TensorShape([])

        # print("self.batch_shape = ", self.batch_shape)
        # print("type(self.batch_shape) = ", type(self.batch_shape) )
        # print("len(self.batch_shape) = ", len(self.batch_shape) )

        # print("self.batch_shape[0] = ", self.batch_shape[:2])
        # print("self.batch_shape[0] = ", self.batch_shape[:2])

        # print("self.event_shape = ", self.event_shape)
        # print("len(self.event_shape) = ", len(self.event_shape))

        # self.event_shape = scale.shape

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
        # var = self.scale**2
        log_pdf = -tf.math.log(self.scale * tf.math.sqrt(2 * tf.constant(np.pi))) - 0.5 * ((x - self.loc) ** 2) / (self.scale ** 2)

        # log_pdf = torch.tensor(log_pdf.numpy())
        
        return log_pdf

    def sample(self, shape=None):
        """
        从正态分布中采样

        Args:
            shape: 采样的形状。如果为 None，默认返回单个样本。

        Returns:
            从正态分布中采样的张量
        """
        print("1sample.shape = ", shape)
        if shape == None or shape == tf.TensorShape([]):
            shape = self.loc.shape
        print("1sample.shape = ", shape)
        sampled = tf.random.normal(shape=shape, mean=self.loc, stddev=self.scale)

        # sampled = torch.tensor(sampled.numpy())

        # print("normal: sampled = ", sampled)

        return sampled

    def entropy(self):
        """
        计算正态分布的熵

        Returns:
            正态分布的熵
        """
        # 使用公式 H(X) = 0.5 * log(2 * pi * e * std^2)
        entropy = 0.5 * tf.math.log(2 * tf.constant(np.pi) * tf.constant(np.e) * self.scale ** 2)
        
        # entropy = torch.tensor(entropy.numpy())

        return entropy


# import tensorflow as tf

def _sum_rightmost(x, n):
    """
    对张量的最后 n 个维度进行求和。
    
    Args:
        x: 输入的张量。
        n: 需要求和的最后 n 个维度的数量。
        
    Returns:
        求和后的张量。
    """
    # 获取张量的总维度数
    num_dims = len(x.shape)
    
    # 求和的维度是从最后一个维度向前数 n 个维度
    axes = list(range(num_dims - n, num_dims))
    
    # 使用 tf.reduce_sum 对指定维度进行求和
    return tf.reduce_sum(x, axis=axes)



class Independent:
    def __init__(self, base_distribution, reinterpreted_batch_ndims, validate_args=None):
        if reinterpreted_batch_ndims > len(base_distribution.batch_shape):
            raise ValueError(
                "Expected reinterpreted_batch_ndims <= len(base_distribution.batch_shape), "
                f"actual {reinterpreted_batch_ndims} vs {len(base_distribution.batch_shape)}"
            )
        shape = base_distribution.batch_shape + base_distribution.event_shape
        # print("shape = ", shape)

        # if base_distribution.event_shape != :
        event_dim = reinterpreted_batch_ndims + len(base_distribution.event_shape)
        # print("event_dim = ", event_dim)
        # print("reinterpreted_batch_ndims = ", reinterpreted_batch_ndims)
        # print("len(base_distribution.event_shape) = ", len(base_distribution.event_shape))

        self.batch_shape = shape[: len(shape) - event_dim]
        self.event_shape = shape[len(shape) - event_dim :]

        # print("self.batch_shape = ", self.batch_shape)
        # print("self.event_shape = ", self.event_shape)

        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        # super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value)
        # print("log_prob before = ", log_prob)
        return _sum_rightmost(log_prob, self.reinterpreted_batch_ndims)

    def entropy(self):
        entropy = self.base_dist.entropy()
        return _sum_rightmost(entropy, self.reinterpreted_batch_ndims)
    
    def sample(self, sample_shape=tf.TensorShape([])):
        return self.base_dist.sample(sample_shape)
    

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
        
        # print("logits.shape = ", logits.shape)

        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )


        if probs is not None:
            self.probs = probs
        elif logits is not None:
            self.logits = logits
            self.probs = tf.nn.softmax(logits, axis=-1)
        # else:
        #     raise ValueError("Must specify either probs or logits.")
    

        # self.batch_shape = logits.shape

        # self.event_shape = tf.TensorShape([])

        # if self.probs is not None:
        if len(self.probs.shape) < 1:
            raise ValueError("`probs` parameter must be at least one-dimensional.")
        # self.probs = probs / probs.sum(-1, keepdim=True)
        if probs is not None:
            self.probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)

        # else:
        #     raise ValueError("must specify probs.")
            # if logits.dim() < 1:
            #     raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            # self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        self._param = self.probs if probs is not None else self.logits
        self._num_events = self._param.shape[-1]
        # print("type(self._num_events) = ", type(self._num_events))
        batch_shape = (
            self._param.shape[:-1] if len(self._param.shape) > 1 else tf.TensorShape([])
        )
        self.batch_shape = batch_shape
        # super().__init__(batch_shape, validate_args=validate_args)


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





class MixtureSameFamily:
    def __init__(
            self, mixture_distribution, component_distribution, validate_args=None
        ):
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution

        if not isinstance(self._mixture_distribution, Categorical):
            raise ValueError(
                " The Mixture distribution needs to be an "
                " instance of torch.distributions.Categorical"
            )

        # if not isinstance(self._component_distribution, Distribution):
        #     raise ValueError(
        #         "The Component distribution need to be an "
        #         "instance of torch.distributions.Distribution"
        #     )

        # Check that batch size matches
        mdbs = self._mixture_distribution.batch_shape
        print("self._component_distribution.batch_shape = ", self._component_distribution.batch_shape)
        cdbs = self._component_distribution.batch_shape[:-1]
        # cdbs = self._component_distribution.batch_shape
        
        for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
            if size1 != 1 and size2 != 1 and size1 != size2:
                raise ValueError(
                    f"`mixture_distribution.batch_shape` ({mdbs}) is not "
                    "compatible with `component_distribution."
                    f"batch_shape`({cdbs})"
                )

        # Check that the number of mixture component matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._component_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError(
                f"`mixture_distribution component` ({km}) does not"
                " equal `component_distribution.batch_shape[-1]`"
                f" ({kc})"
            )
        self._num_component = km

        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)

        self.batch_shape = cdbs
        self.event_shape = event_shape

        # super().__init__(
        #     batch_shape=cdbs, event_shape=event_shape, validate_args=validate_args
        # )


    def log_prob(self, x):
        # if self._validate_args:
        #     self._validate_sample(x)
        x = tf.expand_dims(x, axis=-1 - self._event_ndims)
        log_prob_x = self.component_distribution.log_prob(x)  # [S, B, k]


        log_mix_prob = tf.math.log(self.mixture_distribution.probs)

        return torch_logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]



def torch_logsumexp(input, dim):
    return tf.reduce_logsumexp(input, axis=dim)






def torch_min(input, dim = None, other = None):
    if other == None:
        return tf.reduce_min(input, axis=dim)
    else:
        return tf.minimum(input, other)

def torch_max(input, dim = None, other = None):
    if other == None:
        return tf.reduce_max(input, axis=dim)
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




def torch_tensor(input_numpy_array):
    return tf.convert_to_tensor(input_numpy_array)




def torch_clamp(input, min = float('-inf'), max = float('inf'), out=None):
    out = tf.clip_by_value(input, min, max)
    return out



# torch.zeros(size, dtype=torch.float32, device=None, requires_grad=False)
def torch_zeros(*size, dtype=tf.float32):
    size_list = []
    for cur_size in size:
        if not isinstance(cur_size, int):
            break
        else:
            size_list.append(cur_size)
            
    return tf.zeros(size_list, dtype=tf.float32, name=None)


def torch_ones(*size, dtype=tf.float32):
    size_list = []
    for cur_size in size:
        if not isinstance(cur_size, int):
            break
        else:
            size_list.append(cur_size)
            
    return tf.ones(size_list, dtype=tf.float32, name=None)




def torch_nanmean(input):

    input_no_nan = tf.boolean_mask(input, ~tf.math.is_nan(input))
    result = tf.reduce_mean(input_no_nan)

    return result





def torch_prod(input, dim = None, keepdim=False):
    return tf.reduce_prod(input, axis=dim, keepdims = keepdim)
        





def torch_cat(tensors, dim=0, out=None):
    out =  tf.concat(tensors, axis = dim, name='concat')
    return out






def torch_hstack(input):
    return tf.concat(input, axis=1)





def torch_linspace(start, end, steps):
    return tf.linspace(start, end, num=steps)






def torch_argmax(input, dim=None):
    return tf.argmax(input, axis=dim, output_type=tf.dtypes.int64, name='ArgMax')
    







def torch_tensor_view(input, *args):
    return tf.reshape(input, [*args])






def torch_arange(start, end, step=1, dtype=None, device=None, requires_grad=False):
    return tf.range(start=start, limit=end, delta=step, dtype=dtype, name='range')







def torch_logsumexp(input, dim, keepdim=False, dtype=None):
    return tf.reduce_logsumexp(input, axis=dim, keepdims=keepdim)



def torch_mse_loss(input, target, reduction='mean'):
    mse = tf.reduce_mean(tf.square(input - target))
    return mse




def torch_unsqueeze(input, dim):
    return tf.expand_dims(input, axis=dim)



def torch_squeeze(input, dim=None):
    return tf.squeeze(input, axis=dim)




# torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
def torch_full(size, fill_value, dtype=None):
    result = tf.fill(size, fill_value)
    if dtype != None:
        result = tf.cast(result, dtype=dtype)
    return result






def torch_sqrt(input):
    return tf.sqrt(input)





def torch_tensor_float(input):
    return tf.cast(input, tf.float32)





def torch_tensor_expand(input, *args):
    return tf.broadcast_to(input, [*args])





def torch_triu(input, diagonal=0, *, out=None):
    dim = len(input.shape)
    assert dim == 2, "shape must be 2-D for torch.triu"
    rows, cols = input.shape.as_list()

    row_indices = torch_tensor_expand( torch_tensor_view( torch_arange(0, rows), -1, 1 ), rows, cols)
    col_indices = torch_tensor_expand( torch_tensor_view( torch_arange(0, cols), 1, -1 ), rows, cols)

    mask = col_indices - row_indices >= diagonal

    return tf.where(mask, input, 0)








def torch_round(input):
    return tf.round(input)







def torch_meshgrid(*tensors, indexing=None):
    if indexing == "xy":
        return tf.meshgrid(*tensors, indexing=indexing)

    return tf.meshgrid(*tensors, indexing="ij")






def torch_sum(input, dim, keepdim = False):
    return tf.reduce_sum(input, axis=dim, keepdims = keepdim)









def torch_cumprod(input, dim):
    return tf.math.cumprod(input, axis=dim)








def torch_randn(*size):
    return tf.random.normal([*size])







def torch_randperm(n):
    return tf.random.shuffle(tf.range(n))







def torch_zeros_like(input, dtype=None):
    return tf.zeros_like(input, dtype=dtype)




def torch_full_like(input, fill_value, dtype=None):
    return tf.fill(tf.shape(input), fill_value)




def torch_from_numpy(ndarray):
    return tf.convert_to_tensor(value=ndarray)




def torch_exp(input):
    return tf.math.exp(input)




def torch_item(x):
    out = torch_squeeze(x)
    out = out.numpy().item()
    return out







def torch_repeat_interleave(tensor, repeats, dim=None):

    
    if isinstance(repeats, int):
        # repeats = torch_full_like(torch_tensor(tensor.shape), repeats)  # 扩展为与 tensor 大小一致
        pass
    else:
        raise ValueError("non scalar repeats is not implemented for this function")

    print("repeats = ", repeats)

    cur_tensor = tensor


    if dim == None:
        cur_tensor = torch_flatten(tensor)
        
    result = []

    if len(cur_tensor.shape) == 1:
        for i in range(cur_tensor.shape[0]):
            result.extend( [cur_tensor[i], ] * repeats )
        return tf.convert_to_tensor( np.array(result) )
        
    elif len(tensor.shape) >= 2:

        for i in range(tensor.shape[dim]):
            new_shape = tensor.shape.as_list()
            new_shape[dim] = 1
            row = tf.gather( tensor, i, axis=dim)
            row = tf.reshape(row, new_shape)
            result.extend( [row ] * repeats )        
        return tf.concat( result, axis=dim )
    
    else:
        raise ValueError("tensor.shape > 2 is not implemented for hte repeat_interleave()")











def torch_reshape():
    pass





def torch_randint():
    pass







def torch_vmap():
    pass






def torch_func_stack_module_state():
    pass









def torch_func_functional_call():
    pass



