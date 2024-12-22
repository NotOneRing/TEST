import tensorflow as tf
import numpy as np



def torch_tensor_item(tensor):
    return tensor.numpy().item()


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





def torch_arange(start, end, step, dtype):
    # torch.arange(start=0, end, step=1, *, 
    # out=None, dtype=None, layout=torch.strided, 
    # device=None, requires_grad=False)
    return tf.range(start=start, limit=end, delta=step, dtype=dtype)





def torch_abs(input, *, out=None):
    out = tf.abs(input)
    return out



def torch_square(input, *, out=None):
    out = tf.square(input)
    return out



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







def torch_min(input, dim = None, other = None):
    if other == None:
        if dim == None:
            return tf.reduce_min(input)
        else:
            min_values = tf.reduce_min(input, axis=dim)
            min_indices = tf.math.argmin(input, axis=dim)
            from collections import namedtuple
            MinResult = namedtuple('MinResult', ['values', 'indices'])
            result = MinResult(values=min_values, indices=min_indices)
            return result
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
    assert isinstance(input_numpy_array, np.ndarray), "torch_tensor input type wrong"
    return tf.convert_to_tensor(input_numpy_array)



def torch_clip(input, min = float('-inf'), max = float('inf'), out=None):
    #alias for torch_clamp
    out = tf.clip_by_value(input, min, max)
    return out



def torch_clamp(input, min = float('-inf'), max = float('inf'), out=None):
    out = tf.clip_by_value(input, min, max)
    return out




def torch_log(input):
    tensor = input
    if tensor.dtype == tf.int32 or tensor.dtype == tf.int64:
        tensor = tf.cast(tensor, tf.float32)

    return tf.math.log(tensor)



def torch_tensor_clamp_(input, min = float('-inf'), max = float('inf')):
    input = tf.clip_by_value(input, min, max)








# torch.zeros(size, dtype=torch.float32, device=None, requires_grad=False)
def torch_zeros(*size, dtype=tf.float32):

    if isinstance(size[0], (list, tuple)):
        size_list = size[0]
    else:
        size_list = list(size)
    
    return tf.zeros(size_list, dtype=tf.float32, name=None)






def torch_ones(*size, dtype=tf.float32):
    
    if isinstance(size[0], (list, tuple)):
        size_list = size[0]
    else:
        size_list = list(size)
    
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
    # print("args = ", args)
    if isinstance(args[0], (tuple, list)):
        result = tf.reshape(input, args[0] )
    else:
        result = tf.reshape(input, [*args] )

    return result






def torch_reshape(input, *shape):
    if isinstance(shape[0], (tuple, list)):
        result = tf.reshape(input, shape[0] )
    else:
        result = tf.reshape(input, [*shape] )

    return result






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
    output = tf.cast(input, tf.float32)
    return output



def torch_tensor_long(input):
    return tf.cast(input, tf.int64)




def torch_tensor_expand(input, *args):
    if isinstance(args[0], (tuple, list)):
        result = tf.broadcast_to(input, args[0] )
    else:
        result = tf.broadcast_to(input, [*args] )

    return result


def torch_tensor_expand_as(input, other):
    return tf.broadcast_to(input, other.shape)





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







def torch_meshgrid(*tensors, indexing="ij"):

    if isinstance(tensors[0], (tuple, list)):
        # print("branch1")
        return tf.meshgrid(*tensors[0], indexing=indexing)

    else:
        # print("branch2")
        return tf.meshgrid(*tensors, indexing=indexing)







def torch_sum(input, dim, keepdim = False):
    return tf.reduce_sum(input, axis=dim, keepdims = keepdim)









def torch_cumprod(input, dim):
    return tf.math.cumprod(input, axis=dim)








def torch_randn(*size):
    if isinstance(size[0], (tuple, list)):
        return tf.random.normal(size[0])

    else:
        return tf.random.normal(size)






def torch_randperm(n):
    return tf.random.shuffle(tf.range(n))




def torch_randn_like(input, *, dtype=None
                     # , layout=None 
                     # , device=None
                     # , requires_grad=False
                     # , memory_format=torch.preserve_format
                     ):
    # wrapper for torch.randn_like(x)
    # Returns a tensor with the same size as input 
    # that is filled with random numbers from a normal
    # distribution with mean 0 and variance 1. 
    # Please refer to torch.randn() for the 
    # sampling process of complex dtypes. 
    # torch.randn_like(input) is equivalent 
    # to torch.randn(input.size(), dtype=input.dtype, 
    # layout=input.layout, device=input.device).

    shape = input.shape.as_list()

    if input.dtype.is_floating:
        return torch_randn(*shape)
    else:
        raise RuntimeError("'normal_kernel_cpu' not implemented for 'Long' type.")



def torch_zeros_like(input, dtype=None):
    return tf.zeros_like(input, dtype=dtype)




def torch_full_like(input, fill_value, dtype=None):
    return tf.fill(tf.shape(input), fill_value)




def torch_from_numpy(ndarray):
    return tf.convert_to_tensor(value=ndarray)




def torch_exp(input):
    return tf.math.exp(input)









def torch_flip(input, dims):
    return tf.reverse(input, axis=dims)







def torch_randint(low=0, *, high, size, dtype = None):
    if high is None or size is None:
        raise ValueError("Both 'high' and 'size' must be specified.")
    
    if dtype is None:
        dtype = tf.int32  # Default dtype
    
    # Generate random integers
    return tf.random.uniform(shape=size, minval=low, maxval=high, dtype=dtype)






def torch_tensor_transpose(input, dim0, dim1):
    dim_lens = len(input.shape)
    perm = list(range(dim_lens))
    temp = perm[dim0]
    perm[dim0] = perm[dim1]
    perm[dim1] = temp
    return tf.transpose(input, perm=perm)




def torch_tensor_clone(input):
    return tf.identity(input)





def torch_tensor_masked_fill(tensor, mask, value):
    broadcasted_mask = mask if mask.shape == tensor.shape else torch_tensor_expand(mask, tensor)
    output = torch_tensor_clone(tensor)
    output[broadcasted_mask] = value
    return output










def torch_item(x):
    out = torch_squeeze(x)
    out = out.numpy().item()
    return out






def nn_functional_pad_replicate(x, pad):
    # Extract dimensions
    # batch, height, width, channels = x.shape
    from copy import deepcopy
 
    result = deepcopy(x)
 
    total_dim = len(x.shape)
    assert len(pad) % 2 == 0
    pad_dim = len(pad) / 2


    for i in range(int(pad_dim)):
        pad_left_number = pad[2 * i]
        pad_right_number = pad[2 * i + 1]

        axis = total_dim - i - 1

        left = tf.gather(result, 0, axis=axis)
        right = tf.gather(result, x.shape[axis]-1, axis=axis)

        left = tf.expand_dims(left, axis=axis)
        right = tf.expand_dims(right, axis=axis)

        left = tf.repeat(left, repeats=pad_left_number, axis=axis)
        right = tf.repeat(right, repeats=pad_right_number, axis=axis)
        result = tf.concat([left, result, right], axis=axis)

    return result
 
    # # Pad height (top and bottom)
    # top = tf.repeat(x[:, :, :, :1], repeats=pad, axis=1)  # Replicate the first row `pad` times
    # bottom = tf.repeat(x[:, :, :, -1:], repeats=pad, axis=1)  # Replicate the last row `pad` times
    # x = tf.concat([top, x, bottom], axis=1)

    # # Pad width (left and right)
    # left = tf.repeat(x[:, :, :1, :], repeats=pad, axis=2)  # Replicate the first column `pad` times
    # right = tf.repeat(x[:, :, -1:, :], repeats=pad, axis=2)  # Replicate the last column `pad` times
    # x = tf.concat([left, x, right], axis=2)

    # return x







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











def torch_vmap(func, *parameters, in_dims=0):
    outputs = tf.vectorized_map(func, parameters)
    return outputs







def torch_func_stack_module_state(models):
    # 堆叠所有模型的参数和缓冲区
    trainable = [tf.stack([var for var in model.trainable_variables])
                 for model in models]
    non_trainable = [tf.stack([var for var in model.non_trainable_variables])
                     for model in models]
    return trainable, non_trainable









def torch_func_functional_call(model, params, x):
    # Performs a functional call on the module by replacing the module parameters and buffers with the provided ones.
    former_params = model.trainable_variables

    for var, param in zip(model.trainable_variables, params):
        var.assign(param)
    result = model(x)

    for var, param in zip(model.trainable_variables, former_params):
        var.assign(param)

    return result










def torch_model_train():
    pass



def torch_model_eval():
    pass





















def torch_nn_init_normal_(variable, mean=0.0, std=1.0):
    """
    Mimic torch.nn.init.normal_ to initialize TensorFlow variables with values drawn
    from a normal distribution.
    Args:
        variable: A TensorFlow variable or tensor (tf.Variable or tf.Tensor) to initialize.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
    Returns:
        None: The variable is updated in place.
    """
    if not isinstance(variable, tf.Variable):
        raise ValueError("Input variable must be a tf.Variable.")

    # Draw values from a normal distribution
    normal_values = np.random.normal(loc=mean, scale=std, size=variable.shape)

    # Assign the values to the TensorFlow variable
    variable.assign(normal_values.astype(np.float32))




def torch_nn_init_zeros_(tensor):
    """
    Mimic torch.nn.init.zeros_ to initialize TensorFlow variables with zeros.

    Args:
        variable: A TensorFlow variable or tensor (tf.Variable or tf.Tensor) to initialize.

    Returns:
        None: The variable is updated in place.
    """
    if not isinstance(tensor, tf.Variable):
        raise ValueError("Input variable must be a tf.Variable.")
    
    # Assign zeros to the variable
    tensor.assign(tf.zeros_like(tensor))




def torch_nn_init_ones_(tensor):
    """
    Mimic torch.nn.init.ones_ to initialize TensorFlow variables with ones.

    Args:
        variable: A TensorFlow variable or tensor (tf.Variable or tf.Tensor) to initialize.

    Returns:
        None: The variable is updated in place.
    """
    if not isinstance(tensor, tf.Variable):
        raise ValueError("Input variable must be a tf.Variable.")
    
    # Assign ones to the variable
    tensor.assign(tf.ones_like(tensor))




class nn_TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(nn_TransformerDecoderLayer, self).__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
        self.cross_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_feedforward, activation=activation),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, tgt, memory, tgt_mask=None, memory_mask=None, training=None):
        # Self-attention on target
        tgt2 = self.self_attn(tgt, tgt, attention_mask=tgt_mask, training=training)
        tgt = tgt + self.dropout1(tgt2, training=training)
        tgt = self.norm1(tgt)

        # Cross-attention between target and memory
        tgt2 = self.cross_attn(tgt, memory, attention_mask=memory_mask, training=training)
        tgt = tgt + self.dropout2(tgt2, training=training)
        tgt = self.norm2(tgt)

        # Feedforward network
        tgt2 = self.ffn(tgt, training=training)
        tgt = tgt + self.dropout3(tgt2, training=training)
        tgt = self.norm3(tgt)

        return tgt

class nn_TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, nhead, dim_feedforward, dropout, activation):
        super(nn_TransformerDecoder, self).__init__()
        self.layers = [
            nn_TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(n_layers)
        ]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, tgt, memory, tgt_mask=None, memory_mask=None, training=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, training=training)
        return self.norm(tgt)








class nn_TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(nn_TransformerEncoderLayer, self).__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_feedforward, activation=activation),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training):
        # Self-attention
        attn_output = self.self_attn(x, x, training=training)
        x = x + self.dropout1(attn_output, training=training)
        x = self.norm1(x)

        # Feedforward network
        ffn_output = self.ffn(x, training=training)
        x = x + self.dropout2(ffn_output, training=training)
        x = self.norm2(x)
        return x





class nn_TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, nhead, dim_feedforward, dropout, activation):
        super(nn_TransformerEncoder, self).__init__()
        self.layers = [
            nn_TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(n_layers)
        ]

    def call(self, x, training):
        for layer in self.layers:
            x = layer(x, training=training)
        return x






class nn_Mish(tf.keras.layers.Layer):
    def __init__(self):
        super(nn_Mish, self).__init__()

    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))







# Define TensorFlow nn.ReLU wrapper
class nn_ReLU(tf.keras.layers.Layer):
    def __init__(self, inplace=False):
        super(nn_ReLU, self).__init__()
        self.relu = tf.keras.layers.ReLU()
        self.inplace = inplace
    def call(self, x):
        if self.inplace:
            x = self.relu(x)
            return x
        else:
            return self.relu(x)










class nn_Dropout(tf.keras.layers.Layer):
    def __init__(self, p=0.5):
        super(nn_Dropout, self).__init__()
        self.model = tf.keras.layers.Dropout(rate = p)

    # torch.nn.Dropout(p=0.5, inplace=False)
    # tf.keras.layers.Dropout(
    #     rate, noise_shape=None, seed=None, **kwargs
    # )

    def call(self, net_params):
        return self.model(net_params)














class nn_Linear(tf.keras.layers.Layer):
# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# tf.keras.layers.Dense(
#     units,
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     lora_rank=None,
#     **kwargs
# )
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(nn_Linear, self).__init__()
        self.model = tf.keras.layers.Dense(
            out_features,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            # kernel_regularizer=None,
            # bias_regularizer=None,
            # activity_regularizer=None,
            # kernel_constraint=None,
            # bias_constraint=None,
            # ,**kwargs
        )

    def call(self, x):
        return self.model(x)


    # def __getattr__(self, name):
    #     if hasattr(self.model, name):
    #         return getattr(self.model, name)
    #     else:
    #         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # def __getattr__(self, name):
    #     # 先检查当前对象的属性，避免无限递归
    #     if name in self.__dict__:
    #         return self.__dict__[name]
    #     # 再检查 model 的属性
    #     if hasattr(self.model, name):
    #         return getattr(self.model, name)
    #     # 如果都没有，抛出 AttributeError
    #     raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    # def __getattr__(self, name):
    #     # 避免递归调用：直接访问 __dict__ 或 object.__getattribute__
    #     try:
    #         return object.__getattribute__(self, name)
    #     except AttributeError:
    #         pass

    #     # 检查是否为 model 的属性
    #     model = object.__getattribute__(self, "model")

    #     if hasattr(model, name):
    #         return getattr(model, name)

    #     # 如果仍然找不到，抛出异常
    #     raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")







class nn_LayerNorm(tf.keras.layers.Layer):
# torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)
# tf.keras.layers.LayerNormalization(
#     axis=-1,
#     epsilon=0.001,
#     center=True,
#     scale=True,
#     rms_scaling=False,
#     beta_initializer='zeros',
#     gamma_initializer='ones',
#     beta_regularizer=None,
#     gamma_regularizer=None,
#     beta_constraint=None,
#     gamma_constraint=None,
#     **kwargs
# )
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None):
        super(nn_LayerNorm, self).__init__()

        self.model = tf.keras.layers.LayerNormalization(
            axis=-1,
            epsilon=0.001,
            center=True,
            scale=True,
            rms_scaling=False,
            beta_initializer='zeros',
            gamma_initializer='ones',
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None
            # ,**kwargs
        )   

    def call(self, x):
        return self.model(x)















def nn_Parameter(data=None, requires_grad=True):
    if data is None:
        raise ValueError("data cannot be None. Please provide a tensor value.")
    return tf.Variable(data, trainable=requires_grad, name="nn_parameter")










# Define TensorFlow nn.Sequential wrapper
class nn_Sequential(tf.keras.layers.Layer):
    def __init__(self, *args):
        super(nn_Sequential, self).__init__()
        # print("len(args) = ", len(args))
        # print("args = ", args)

        self.model_list = []

        if isinstance(args[0], (tuple, list)):
            for module in args[0]:
               self.model_list.append(module)

        else:
            for module in args:
                self.model_list.append(module)

    def call(self, x):
        output = x
        for module in self.model_list:
            output = module(output)
        return output
    
    def __getitem__(self, id):
        # print("getitem: len(self.model_list) = ", len(self.model_list))
        return self.model_list[id]

    def __iter__(self):
        return iter(self.model_list)





class nn_ModuleList(tf.keras.layers.Layer):
    def __init__(self, modules=None):
        super(nn_ModuleList, self).__init__()
        self.modules = []
        if modules is not None:
            for module in modules:
                self.append(module)

    def append(self, module):
        if not isinstance(module, tf.keras.layers.Layer):
            raise ValueError("All modules must be instances of tf.keras.layers.Layer")
        self.modules.append(module)
        # Automatically track the layer by assigning it to an attribute
        setattr(self, f"module_{len(self.modules) - 1}", module)

    def extend(self, modules):
        for module in modules:
            self.append(module)

    def call(self, x):
        outputs = []
        output = x
        for module in self.modules:
            output = module(output)
            outputs.append(output)
        return outputs


    def __getitem__(self, idx):
        return self.modules[idx]

    def __len__(self):
        return len(self.modules)

    def __repr__(self):
        return f"nn_ModuleList({self.modules})"









# class nn_Embedding(tf.keras.layers.Layer):
#     # torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, 
#     # norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)    
#     def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, 
#                  scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
#         super(nn_Embedding, self).__init__()
#         pass
    
#     def call(self, x):
#         pass



class nn_Embedding(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, 
                 scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
        """
        A TensorFlow wrapper to replicate the functionality of torch.nn.Embedding.
        
        Args:
            num_embeddings (int): Size of the embedding dictionary.
            embedding_dim (int): Size of each embedding vector.
            padding_idx (int, optional): Specifies padding index. Embeddings for this index are always zero.
            max_norm (float, optional): If given, will renormalize embeddings to have a norm less than this value.
            norm_type (float, optional): The p-norm to compute for the max_norm option. Default is 2.0.
            scale_grad_by_freq (bool, optional): If True, scale gradients by inverse of word frequency.
            sparse (bool, optional): Not used in TensorFlow, included for API compatibility.
            _weight (np.ndarray, optional): Predefined weight matrix for embeddings.
            _freeze (bool, optional): If True, the embedding weights are frozen and not updated during training.
            device, dtype: Not used, included for API compatibility.
        """
        super(nn_Embedding, self).__init__(dtype=dtype)
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self._freeze = _freeze

        # Initialize embedding weights
        if _weight is not None:
            self.embeddings = tf.Variable(_weight, trainable=not _freeze, dtype=self.dtype)
        else:
            initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
            self.embeddings = tf.Variable(
                initializer([num_embeddings, embedding_dim]), trainable=not _freeze, dtype=self.dtype
            )

        # Ensure padding_idx embeddings are always zero
        if self.padding_idx is not None:
            self.embeddings[self.padding_idx].assign(tf.zeros([embedding_dim], dtype=self.dtype))

    def call(self, x):
        """
        Args:
            x (Tensor): Indices of the embeddings to retrieve.
        
        Returns:
            Tensor: The embedding vectors corresponding to input indices.
        """
        # Gather embeddings
        embedded = tf.nn.embedding_lookup(self.embeddings, x)

        # Apply max_norm constraint if specified
        if self.max_norm is not None:
            norms = tf.norm(embedded, ord=self.norm_type, axis=-1, keepdims=True)
            embedded = tf.where(
                norms > self.max_norm,
                embedded * (self.max_norm / norms),
                embedded
            )

        return embedded




































class nn_LayerNorm(tf.keras.layers.Layer):
    # torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)
    def __init__(self, normalized_shape, epsilon=1e-5):
        """
        A wrapper for PyTorch's nn.LayerNorm in TensorFlow.
        Args:
            normalized_shape (int or tuple): Input shape for layer normalization.
            epsilon (float): A small value to add to the denominator for numerical stability.
        """
        super(nn_LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

        # Define trainable parameters gamma (scale) and beta (offset)
        self.gamma = self.add_weight(
            name="gamma",
            shape=self.normalized_shape,
            initializer="ones",
            trainable=True
        )
        self.beta = self.add_weight(
            name="beta",
            shape=self.normalized_shape,
            initializer="zeros",
            trainable=True
        )

    def call(self, x):
        """
        Forward pass for LayerNorm.

        Args:
            x (tf.Tensor): Input tensor to normalize.

        Returns:
            tf.Tensor: The normalized tensor.
        """
        print("type(normalized_shape) = ", type(self.normalized_shape))
        if isinstance(self.normalized_shape, int):
            dims = 1
        else:
            dims = len(self.normalized_shape)
        dim_list = []
        for i in range(-dims, 0, 1):
            dim_list.append(i)
        mean = tf.reduce_mean(x, axis=dim_list, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=dim_list, keepdims=True)
        normalized_x = (x - mean) / tf.sqrt(variance + self.epsilon)

        return self.gamma * normalized_x + self.beta







class nn_MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super(nn_MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        # 线性变换用于 Query, Key 和 Value
        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)

        # 输出变换
        self.output_dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """将最后一个维度切分为多个头"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """计算缩放点积注意力"""
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]

        # 对 Q, K, V 进行线性变换并分割成多个头
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 缩放点积注意力
        output, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)

        # 拼接多个头
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        attention_weights = tf.reduce_mean(attention_weights, axis=1)  # 平均所有头

        # 输出变换
        output = self.output_dense(output)
        return output, attention_weights









def torch_tensor_detach(tensor):
    output_tensor = tf.stop_gradient(tensor)
    return output_tensor









# 定义自定义 torch_std 函数
def torch_std(input, dim=None, *, correction=1, keepdim=False, out=None):
    assert out == None, "Tensor is immutable in TensorFlow, but mutable in PyTorch"

    # 计算均值
    mean = tf.reduce_mean(input, axis=dim, keepdims=True)

    # 计算方差
    variance = tf.reduce_mean(tf.square(input - mean), axis=dim, keepdims=keepdim)

    # 应用 Bessel's 修正
    if correction != 0:
        count = tf.shape(input)[dim] if dim is not None else tf.size(input)
        count = tf.cast(count, tf.float32)
        variance *= count / (count - correction)

    return tf.sqrt(variance)










def torch_tensor_repeat(tensor, *repeats):
    """
    Mimics the behavior of PyTorch's torch.Tensor.repeat in TensorFlow.

    Args:
        tensor (tf.Tensor): The input tensor.
        *repeats: The number of times to repeat along each dimension.

    Returns:
        tf.Tensor: The repeated tensor.
    """

    # print("repeats = ", repeats)

    if not isinstance(tensor, tf.Tensor):
        raise TypeError("Input must be a TensorFlow tensor.")
    if not repeats:
        raise ValueError("At least one repeat value must be provided.")

    # processed_repeats = []
    if isinstance(repeats[0], (tuple, list)):
        repeat_shape = [ *repeats[0] ]
        repeats_tensor = tf.constant(repeats[0], dtype=tf.int32)
    else:
        repeat_shape = [*repeats]
        repeats_tensor = tf.constant(repeats, dtype=tf.int32)

    # Compute the target shape for tiling
    tensor_shape = tf.shape(tensor)


    tensor_dim = len(tensor_shape)
    repeat_dim = len(repeat_shape)

    temp_tensor = tensor

    if repeat_dim > tensor_dim:
        tensor_shape = [1] * (repeat_dim - tensor_dim) + tensor_shape.numpy().tolist()
        temp_tensor = tf.reshape(tensor, tensor_shape)

    # Perform tiling
    repeated_tensor = tf.tile(temp_tensor, repeats_tensor)
    return repeated_tensor













def torch_unravel_index(indices, shape):
    """
    TensorFlow equivalent of torch.unravel_index.

    Args:
        indices (Tensor): 1D tensor of linear indices.
        shape (tuple or list): Shape of the target tensor.

    Returns:
        tuple: A tuple of Tensors representing the unraveled indices for each dimension.
    """
    # indices = tf.convert_to_tensor(indices, dtype=tf.int32)
    # shape = tf.convert_to_tensor(shape, dtype=tf.int32)

    unravel_indices = []
    for dim in reversed(shape):
        unravel_indices.append(indices % dim)
        indices = indices // dim

    return tuple(reversed(unravel_indices))









def torch_register_buffer(self, input, name):
    result = tf.constant(input)
    setattr(self, name, result)








# TensorFlow wrap of torch.split
def torch_split(tensor, split_size_or_sections, dim=0):
    # torch.split(tensor, split_size_or_sections, dim=0)
    # tf.split(value, num_or_size_splits, axis=0, num=None, name='split')
    final_num_or_size_splits_list = []

    if not isinstance(split_size_or_sections, (tuple, list)):
        tensor_dim = tensor.shape[dim]
        import math
        total_splits_number = math.ceil(tensor_dim / split_size_or_sections)
        residual = tensor_dim % split_size_or_sections
        
        if residual > 0:
            for i in range(total_splits_number - 1):
                final_num_or_size_splits_list.append(split_size_or_sections)

            final_num_or_size_splits_list.append(residual)
        else:
            for i in range(total_splits_number):
                final_num_or_size_splits_list.append(split_size_or_sections)
    else:
        final_num_or_size_splits_list = split_size_or_sections

    return tf.split(value=tensor, num_or_size_splits=final_num_or_size_splits_list, axis=dim)








def torch_rand(*size, dtype=tf.dtypes.float32):
    # torch.rand(*size, *, generator=None, out=None, dtype=None, 
    # layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
    # tf.random.uniform(
    #     shape,
    #     minval=0,
    #     maxval=None,
    #     dtype=tf.dtypes.float32,
    #     seed=None,
    #     name=None
    # )

    print("size = ", size)

    if isinstance(size[0], (tuple, list)):
        final_size = [ *size[0] ]
    else:
        final_size = [*size]

    return tf.random.uniform(shape=final_size, dtype=dtype)








# tf.keras.optimizers.Adam(
#     learning_rate=0.001,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-07,
#     amsgrad=False,
#     weight_decay=None,
#     clipnorm=None,
#     clipvalue=None,
#     global_clipnorm=None,
#     use_ema=False,
#     ema_momentum=0.99,
#     ema_overwrite_frequency=None,
#     loss_scale_factor=None,
#     gradient_accumulation_steps=None,
#     name='adam',
#     **kwargs
# )
# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, 
# *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
class torch_optim_Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        A TensorFlow implementation of torch.optim.Adam.

        Args:
            params (list): List of TensorFlow variables to optimize.
            lr (float): Learning rate.
            betas (tuple): Coefficients used for computing running averages of gradient and its square.
            eps (float): Term added to the denominator to improve numerical stability.
            weight_decay (float): Weight decay (L2 penalty).
        """
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # TensorFlow Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=betas[0], beta_2=betas[1], epsilon=eps
        )

    def zero_grad(self):
        """No-op function for compatibility, gradients are reset automatically in TensorFlow."""
        pass


    def step(self, gradients):
        """Apply gradients to parameters."""
        self.optimizer.apply_gradients(zip(gradients, self.params))


    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.params))






# torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, 
# *, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)
class torch_optim_AdamW:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        A TensorFlow implementation of torch.optim.AdamW.

        Args:
            params (list): List of TensorFlow variables to optimize.
            lr (float): Learning rate.
            betas (tuple): Coefficients used for computing running averages of gradient and its square.
            eps (float): Term added to the denominator to improve numerical stability.
            weight_decay (float): Weight decay (L2 penalty).
        """
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # TensorFlow AdamW optimizer
        self.optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr, beta_1=betas[0], beta_2=betas[1], epsilon=eps, weight_decay=weight_decay
        )

    def zero_grad(self):
        """No-op function for compatibility, gradients are reset automatically in TensorFlow."""
        pass

    def step(self, gradients):
        """Apply gradients to parameters."""
        self.optimizer.apply_gradients(zip(gradients, self.params))


    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.params))









def model_forward_backward_gradients(input_features, target_label, loss_func, model):
    # torch_inputs = torch.tensor(inputs)
    # torch_targets = torch.tensor(targets)

    # torch_optimizer.zero_grad()
    # torch_outputs = torch_model(torch_inputs)
    # torch_loss = torch_loss_fn(torch_outputs, torch_targets)
    # torch_loss.backward()

    # torch_optimizer.step()

    inputs = input_features
    targets = target_label
    tf_loss_fn = loss_func
    tf_model = model

    # TensorFlow
    with tf.GradientTape() as tape:
        tf_outputs = tf_model(inputs)
        tf_loss = tf_loss_fn(targets, tf_outputs)
    tf_gradients = tape.gradient(tf_loss, tf_model.trainable_variables)

    return tf_loss, tf_gradients







# def torch_nn_utils_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
#     pass


def torch_nn_utils_clip_grad_norm_and_step(parameters, optimizer, max_norm, grads, norm_type=2.0, error_if_nonfinite=False):
    # torch.nn.utils.clip_grad_norm_
    """
    这里多了一个grads参数，因为tensorflow的grads要紧跟着tf.GradientTape
    模仿 PyTorch 中的 clip_grad_norm_ 函数，裁剪 TensorFlow 模型参数的梯度
    :param grads: 梯度列表
    :param max_norm: 梯度的最大范数
    :param norm_type: 范数类型，默认为 2 范数
    :param error_if_nonfinite: 如果梯度包含非有限值（如 NaN 或 Inf），是否抛出错误
    :return: 裁剪后的梯度
    """
    if norm_type != 2.0:
        raise NotImplementedError("Only L2 norm is currently supported")

    # 计算所有梯度的范数
    grads_finite = [tf.clip_by_value(g, -1e7, 1e7) if g is not None else tf.zeros_like(parameters[i]) for i, g in enumerate(grads)]
    # print("grads_finite = ", grads_finite)

    global_norm = tf.norm(tf.stack([tf.norm(grad) for grad in grads_finite if grad is not None]))

    # print("global_norm = ", global_norm)

    # 如果范数超过最大值，进行裁剪
    clip_coef = max_norm / (global_norm + 1e-6)
    # print("clip_coef = ", clip_coef)
    clip_coef_bf = tf.where(global_norm < max_norm, tf.ones_like(clip_coef), clip_coef)
    # print("clip_coef_bf = ", clip_coef_bf)
    clipped_grads = [grad * clip_coef_bf if grad is not None else None for grad in grads]

    # print("clipped_grads = ", clipped_grads)

    # 如果 error_if_nonfinite 为 True，检查非有限值
    if error_if_nonfinite:
        for g in grads:
            if g is not None and (tf.reduce_any(tf.is_nan(g)) or tf.reduce_any(tf.is_inf(g))):
                raise ValueError("Gradients contain non-finite values.")


    optimizer.apply_gradients(clipped_grads)
    
    return clipped_grads














def torch_tensor_requires_grad_(tensor, requires_grad=True):
    # torch.tensor.requires_grad_
    tensor.trainable = requires_grad
    return tensor














def torch_utils_data_DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False):
    # torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
    #    batch_sampler=None, num_workers=0, collate_fn=None,
    #    pin_memory=False, drop_last=False, timeout=0,
    #    worker_init_fn=None, *, prefetch_factor=2,
    #    persistent_workers=False)
    pass














def torch_no_grad():
    return tf.GradientTape(watch_accessed_variables=False, persistent=True)













def torch_tensor_cpu(tensor):

    return 










# def torch_tensor_to():
#     pass





# def torch_save(obj, f):
#     # torch.save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True)
#     assert isinstance(obj, dict), "input should be a dict"
#     for key, value in obj.items():
#         if isinstance(value, (tf.keras.layers.Layer, tf.keras.Model)):
#             print("save model: key = ", key)
#             value.save(f + "key" + ".h5")
#         elif "optimizer" in key or "lr_scheduler" in key:
#             print("do not save optimizer or others: key = ", key)
#             pass
#         else:
#             pass
#     pass












# def torch_load(network_path, map_location=None, weights_only=False):
#     # torch.load(f, map_location=None, pickle_module=pickle, *, weights_only=False, mmap=None, **pickle_load_args)
    
#     pass





# def torch_load_state_dict():
#     pass

# # checkpoint = torch.load(
# #     network_path,
# #     map_location=self.device,
# #     weights_only=True,
# # )
# # self.load_state_dict(
# #     checkpoint["model"],
# #     strict=True,
# # )



















class tf_CosineAnnealingWarmupRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, first_cycle_steps, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=0, gamma=1.0, last_epoch = -1):
        assert warmup_steps < first_cycle_steps

        super(tf_CosineAnnealingWarmupRestarts, self).__init__()
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr

        self.base_max_lr = max_lr

        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch

        self.base_lr = self.min_lr

        self.last_epoch = last_epoch

        self.lr = self.base_lr

        self.step()

    def __call__(self, step):
        return self.lr
        
    # def __call__(self, epoch = None):
    def step(self, epoch = None):
        import math
        print("tf_CosineAnnealingWarmupRestarts.__call__()")
        print("tf: epoch = ", epoch)

        print("tf: type(epoch) = ", type(epoch))
        print("tf: epoch = ", epoch)

        print("tf: self.last_epoch = ", self.last_epoch)

        #Because tensorflow automatically set epoch for each epoch，to achieve the same optimizer as the pytorch version, we choose to fix epoch=None manually
        if epoch is not None:
            epoch = int(epoch)  # 强制转换为整数

        print("tf: 2epoch = ", epoch)
        epoch = None
        print("tf: 3epoch = ", epoch)

        print("tf: self.base_lr = ", self.base_lr)

        if epoch is None:
            print("tf: step: branch1")
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                print("tf: step: branch1-1")
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            print("tf: step: branch2")
            if epoch >= self.first_cycle_steps:
                print("tf: step: branch2-1")
                if self.cycle_mult == 1.0:
                    print("tf: step: branch2-1-1")
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    print("tf: step: branch2-1-2")
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n

                    print("tf: self.cycle = ", self.cycle)

                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )

                    print("tf: self.step_in_cycle = ", self.step_in_cycle)

                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )

                    print("tf: self.cur_cycle_steps = ", self.cur_cycle_steps)

            else:
                print("tf: step: branch2-2")
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)


        if self.step_in_cycle == -1:
            # return self.base_lrs
            print("tf: get_lr: branch1")
            self.lr = self.base_lr
        elif self.step_in_cycle < self.warmup_steps:
            print("tf: get_lr: branch2")
            self.lr = (self.max_lr - self.base_lr) * self.step_in_cycle / self.warmup_steps + self.base_lr
            # [
                # for base_lr in self.base_lrs
            # ]
        else:
            print("tf: get_lr: branch3")
            self.lr = self.base_lr + (self.max_lr - self.base_lr) * ( 1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                ) / 2

        # 更新学习率
        # tf.keras.backend.set_value(self.optimizer.lr, lr)

        print("tf: lr = ", self.lr)

        return self.lr








def torch_nn_functional_grid_sample(image, grid, mode='bilinear', padding_mode="zeros", align_corners=False):


    def grid_sampler_unnormalize_tf(coord, side, align_corners):
        if align_corners:
            return ((coord + 1) / 2) * (side - 1)
        else:
            return ((coord + 1) * side - 1) / 2
            
    def grid_sampler_compute_source_index_tf(coord, size, align_corners):
        return grid_sampler_unnormalize_tf(coord, size, align_corners)

    def safe_get_tf(image, n, c, x, y, H, W):
        value = tf.zeros([1])
        x = tf.cast(x, tf.int32)  # Ensure x is an integer type
        y = tf.cast(y, tf.int32)  # Ensure y is an integer type
        if x >= 0 and x < W and y >= 0 and y < H:
            value = image[n, c, y, x]
        return value


    assert padding_mode == "zeros", "only zeros padding_mode is implemented right now"
    assert padding_mode == "zeros", "only zeros padding_mode is implemented right now"
    assert mode == "bilinear", "only bilinear is implemented right now"
    assert len(image.shape) == 4, "len(input.shape) must be 4"

    N, H, W, C = image.shape  # TensorFlow uses NHWC format
    grid_H = grid.shape[1]
    grid_W = grid.shape[2]
    
    # output_tensor = tf.zeros_like(image)
    output_tensor = np.zeros_like(image)
    for n in range(N):
        for w in range(grid_W):
            for h in range(grid_H):
                # Get corresponding grid x and y
                x = grid[n, h, w, 1]
                y = grid[n, h, w, 0]
                
                # Unnormalize with align_corners condition
                ix = grid_sampler_compute_source_index_tf(x, W, align_corners)
                iy = grid_sampler_compute_source_index_tf(y, H, align_corners)
                
                x0 = tf.floor(ix)
                x1 = x0 + 1

                y0 = tf.floor(iy)
                y1 = y0 + 1
    
                # Get W matrix before I matrix, as I matrix requires Channel information
                wa = (x1 - ix) * (y1 - iy)
                wb = (x1 - ix) * (iy - y0)
                wc = (ix - x0) * (y1 - iy)
                wd = (ix - x0) * (iy - y0)
                
                # Get values of the image by provided x0, y0, x1, y1 by channel
                for c in range(C):
                    Ia = safe_get_tf(image, n, c, y0, x0, H, W)
                    Ib = safe_get_tf(image, n, c, y1, x0, H, W)
                    Ic = safe_get_tf(image, n, c, y0, x1, H, W)
                    Id = safe_get_tf(image, n, c, y1, x1, H, W)
                    out_ch_val = Ia * wa + Ib * wb + Ic * wc + Id * wd

                    # output_tensor[n, h, w, c] = out_ch_val
                    # output_tensor[n, h, w, c] = out_ch_val.numpy()
                    output_tensor[n, c, h, w] = out_ch_val.numpy()
    output_tensor = tf.convert_to_tensor(output_tensor)
    return output_tensor

































