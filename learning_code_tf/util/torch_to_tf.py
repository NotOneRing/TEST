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



def torch_logsumexp(input, dim):
    return tf.reduce_logsumexp(input, axis=dim)






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



def torch_tensor_clamp_(input, min = float('-inf'), max = float('inf')):
    input = tf.clip_by_value(input, min, max)


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
    output = tf.cast(input, tf.float32)
    return output



def torch_tensor_long(input):
    return tf.cast(input, tf.int64)




def torch_tensor_expand(input, *args):
    return tf.broadcast_to(input, [*args])


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






def torch_reshape(input, *shape):
    return tf.reshape(input, [*shape])








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










class nn_Sequential(tf.keras.layers.Layer):
    def __init__(self, *args):
        super(nn_Sequential, self).__init__()
        self.model_list = []
        for module in args:
            self.model_list.append(module)
        pass
    
    def call(self, x):
        output = x
        for module in self.model_list:
            output = module(output)
        return output



































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
    tensor = tf.stop_gradient(tensor)
    return tensor







def torch_repeat():
    pass





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
def torch_optim_Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, 
                     amsgrad=False, *, foreach=None, maximize=False, capturable=False, 
                     differentiable=False, fused=None):
    pass









def torch_optim_AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, 
                      amsgrad=False, *, maximize=False, foreach=None, capturable=False, 
                      differentiable=False, fused=None):
    
    return  tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=lr,
                first_decay_steps=cfg.train.lr_scheduler.first_cycle_steps,
                t_mul=1.0,
                alpha=cfg.train.lr_scheduler.min_lr / cfg.train.learning_rate,
            ),
        )








def torch_repeat():
    pass







def torch_std(input, dim=None, *, correction=1, keepdim=False, out=None):
    pass








def torch_nn_utils_clip_grad_norm_():
    # torch.nn.utils.clip_grad_norm_
    pass








def torch_utils_data_DataLoader():
    # torch.utils.data.DataLoader
    pass







def torch_rand():
    # torch.rand
    pass






def torch_tensor_to():
    pass









def torch_tensor_requires_grad_():
    # torch.tensor.requires_grad_
    pass





def torch_unravel_index():
    pass





def torch_split():
    pass






def with_torch_no_grad():
    # with torch.no_grad():
    pass










def torch_optimizer_step(optimizer, gradients, parameters):
    # return optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return optimizer.apply_gradients(zip(gradients, parameters))
# self.critic_optimizer.zero_grad()
# loss_critic.backward()
# self.critic_optimizer.step()








def torch_register_buffer(self, input, name):
    result = tf.constant(input)
    setattr(self, name, result)










def torch_tensor_cpu(tensor):
    return 





def torch_save():
    # torch.save
    pass





def torch_load():
    # torch.load()
    pass
