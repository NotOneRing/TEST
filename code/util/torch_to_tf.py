import tensorflow as tf
import numpy as np
from collections import OrderedDict

# tf.random.set_seed(42)


from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, OUTPUT_POSITIONS, OUTPUT_FUNCTION_HEADER




def torch_tensor_permute(input, *dims):
    "A wrapper for torch.Tensor.permute() function"
    if isinstance(dims[0], (tuple, list)):
        result = tf.transpose(input, perm = dims[0] )
    else:
        result = tf.transpose(input, perm = [*dims] )

    return result




def torch_tensor_item(tensor):
    "A wrapper for torch.Tensor.item() function"
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

    assert len( input_tensor.shape.as_list() ) == len( index_tensor.shape.as_list() ), "input_tensor.shape is not equal to index_tensor.shape"

    index_array = index_tensor.numpy()

    input_array = input_tensor.numpy()


    # input_dim_list = input_tensor.shape.as_list()
    # dim_list = index_tensor.shape.as_list()

    input_dim_list = list(input_tensor.shape)
    dim_list = list(index_tensor.shape)

    # print("list(input_tensor.shape) = ", list(input_tensor.shape) )
    # print("list(index_tensor.shape) = ", list(index_tensor.shape) )

    #transfer negative index to positive one
    # print("dim = ", dim)
    # print("input_dim_list = ")
    dim = list(range(len( input_dim_list )))[dim]

    for i in range(len( input_dim_list )):
        if i == dim:
            continue
        if input_dim_list[i] < dim_list[i]:
            raise ValueError(f"Size does not match at dimension {i} expected index {dim_list} to be smaller than self {input_dim_list} apart from dimension { dim }")


    dim_number = len(input_tensor.shape)

    cur_index = [0] * dim_number
    
    output_matrix = np.zeros(dim_list, dtype=np.float32)

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




def torch_quantile(input_tensor, q, dim=None, interpolation='linear'):
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









def torch_min(input, dim = None, other = None):
    if isinstance(dim, (tf.Tensor, tf.Variable)):
        temp = dim
        dim = other
        other = temp
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
    if isinstance(dim, (tf.Tensor, tf.Variable)):
        temp = dim
        dim = other
        other = temp
    if other == None:
        # return tf.reduce_max(input, axis=dim)
        if dim == None:
            return tf.reduce_max(input)
        else:
            max_values = tf.reduce_max(input, axis=dim)
            max_indices = tf.math.argmax(input, axis=dim)
            from collections import namedtuple
            MaxResult = namedtuple('MaxResult', ['values', 'indices'])
            result = MaxResult(values=max_values, indices=max_indices)
            return result
    else:
        return tf.maximum(input, other)


def torch_mean(input_tensor1, dim = None, keepdim = False):
    return tf.reduce_mean(input_tensor1, axis = dim, keepdims = keepdim)



def torch_softmax(input_tensor, dim):
    return tf.nn.softmax(input_tensor, axis=dim)


def torch_stack(tensor_list_to_stack, dim = 0):
    return tf.stack(tensor_list_to_stack, axis = dim)


def torch_multinomial(input, num_samples, replacement = True):
    assert replacement == True, "replacement must be True to use tf.random.categorical"
    return tf.random.categorical(input, num_samples=num_samples)




def torch_where(index_tensor, input_tensor = None, replace_value = None):
    if input_tensor != None and replace_value != None:
        result = tf.where(index_tensor, input_tensor, replace_value)
    else:
        result = tf.where(index_tensor, input_tensor, replace_value)
        assert len(result.shape) == 2, "result reshape must be two"
        true_num = result.shape[0]
        result_dim = result.shape[1]
        if OUTPUT_VARIABLES:
            print("true_num = ", true_num)
            print("result_dim = ", result_dim)
        result_list = []
        for i in range(result_dim):
            gather_result = tf.gather(result, i, axis=1)
            if OUTPUT_VARIABLES:
                print("gather_result = ", gather_result)
            result_list.append(gather_result)
        result = tuple(result_list)
    return result




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
    if isinstance(input, tf.Variable):
        temp_variable = tf.clip_by_value(input, min, max)
        # print("temp_variable = ", temp_variable)
        # print("input = ", input)
        input.assign( temp_variable )
        # print("after input.assign input= ", input)
    # wrong path
    # elif isinstance(input, tf.Tensor):
    #     variable = tf.Variable(input)
    #     variable = tf.clip_by_value(variable, min, max)
    #     tensor_from_variable = tf.convert_to_tensor(variable)
    #     input = None
    #     input = tensor_from_variable
    else:
        raise RuntimeError("Input must be tf.Variable to be able to changed")








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
    first_result = tf.square(input - target)
    if reduction == "none":
        return first_result
    elif reduction == "mean":
        return tf.reduce_mean(first_result)
    elif reduction == "sum":
        return tf.reduce_sum(first_result)




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





def torch_tensor_byte(input):
    output = tf.cast(input, tf.uint8)
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







def torch_sum(input, dim = None, keepdim = False):
    return tf.reduce_sum(input, axis=dim, keepdims = keepdim)









def torch_cumprod(input, dim):
    return tf.math.cumprod(input, axis=dim)








def torch_randn(*size):
    if isinstance(size[0], (tuple, list)):
        return tf.random.normal(size[0])

    else:
        return tf.random.normal(size)






def torch_randperm(n):
    if DEBUG:
        return tf.convert_to_tensor( np.random.permutation(n) )
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


def torch_ones_like(input, dtype=None):
    return tf.ones_like(input, dtype=dtype)



def torch_full_like(input, fill_value, dtype=None):
    return tf.fill(tf.shape(input), fill_value)




def torch_from_numpy(ndarray):
    return tf.convert_to_tensor(value=ndarray)




def torch_exp(input):
    x = tf.cast(input, tf.float32)
    return tf.math.exp(x)









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















def nn_functional_pad(x, pad, mode='replicate'):
    assert mode=="replicate", "only replicate is implemented right now"
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
        # repeats = torch_full_like(torch_tensor(tensor.shape), repeats)  # expand to the same shape as tensor
        pass
    else:
        raise ValueError("non scalar repeats is not implemented for this function")

    if OUTPUT_VARIABLES:
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
        raise ValueError("tensor.shape > 2 is not implemented for the repeat_interleave()")




def torch_tensor_detach(tensor):
    output_tensor = tf.stop_gradient(tensor)
    return output_tensor









# Define custom torch_std function
def torch_std(input, dim=None, *, correction=1, keepdim=False, out=None):
    assert out == None, "Tensor is immutable in TensorFlow, but mutable in PyTorch"

    # Calculate mean
    mean = tf.reduce_mean(input, axis=dim, keepdims=True)

    # Calculate variance
    variance = tf.reduce_mean(tf.square(input - mean), axis=dim, keepdims=keepdim)

    # Apply Bessel's correction
    if correction != 0:
        count = tf.shape(input)[dim] if dim is not None else tf.size(input)
        count = tf.cast(count, tf.float32)
        variance *= count / (count - correction)

    # Calculate standard deviation
    result = tf.sqrt(variance)
    result = tf.cast(result, tf.float32)
    return result










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

    print("torch_tensor_repeat: 1")

    # processed_repeats = []
    if isinstance(repeats[0], (tuple, list)):
        repeat_shape = [ *repeats[0] ]
        repeats_tensor = tf.constant(repeats[0], dtype=tf.int32)
    else:
        repeat_shape = [*repeats]
        repeats_tensor = tf.constant(repeats, dtype=tf.int32)

    repeats_tensor = torch_reshape( repeats_tensor, -1)

    print("torch_tensor_repeat: 2")

    # Compute the target shape for tiling
    tensor_shape = tf.shape(tensor)


    tensor_dim = len(tensor_shape)
    repeat_dim = len(repeat_shape)

    print("torch_tensor_repeat: 3")

    temp_tensor = tensor

    if repeat_dim > tensor_dim:
        tensor_shape = [1] * (repeat_dim - tensor_dim) + tensor_shape.numpy().tolist()
        temp_tensor = tf.reshape(tensor, tensor_shape)

    print("torch_tensor_repeat: 4")

    print("temp_tensor = ", temp_tensor)
    print("repeats_tensor = ", repeats_tensor)

    # Perform tiling
    repeated_tensor = tf.tile(temp_tensor, repeats_tensor)

    print("torch_tensor_repeat: 5")

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





def torch_tanh(input):
    return tf.math.tanh(input)





def torch_atanh(input):
    # The domain of the inverse hyperbolic tangent is (-1, 1) 
    # and values outside this range will be mapped to NaN, 
    # except for the values 1 and -1 for which the output 
    # is mapped to +/-INF respectively.
    return tf.math.atanh(input)









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

    if OUTPUT_VARIABLES:
        print("size = ", size)

    if isinstance(size[0], (tuple, list)):
        final_size = [ *size[0] ]
    else:
        final_size = [*size]

    return tf.random.uniform(shape=final_size, dtype=dtype)






def torch_dot(input, tensor, *, out=None):
    assert len(input.shape) == 1, "len(input.shape) must be 1"
    assert len(tensor.shape) == 1, "len(tensor.shape) must be 1"
    out = tf.reduce_sum(input * tensor)
    return out





def torch_vmap(func, *inputs, in_dims=0, out_dims=0):
    # print("inputs = ", inputs)
    # print("type(inputs) = ", type(inputs) )

    # print("*inputs = ", *inputs)
    # print("type(*inputs) = ", type(*inputs))
    out = []
    for tensor in inputs:
        cur_tensor = torch_tensor_clone(tensor)
        out.append(cur_tensor)
    
    if OUTPUT_VARIABLES:
        print("out = ", out)

    if in_dims != 0:
        for i, tensor in enumerate(out):
            out[i] = torch_tensor_transpose(out, 0, in_dims)

    out = tuple(out)

    if OUTPUT_VARIABLES:
        print("out = ", out)


    outputs_tf = tf.vectorized_map(lambda out: torch_dot(out[0], out[1]), out)
    
    if OUTPUT_VARIABLES:
        print("outputs_tf = ", outputs_tf)

    
    if out_dims != 0:
        outputs = torch_tensor_transpose(outputs, 0, out_dims)

    return outputs










def torch_func_stack_module_state(models):
    # Stack all model parameters and buffers
    # trainable = [tf.stack([var for var in model.trainable_variables])
    #              for model in models]
    # non_trainable = [tf.stack([var for var in model.non_trainable_variables])
    #                  for model in models]

    # print("trainable = ", trainable)
    # print("non_trainable = ", non_trainable)

    trainable = {}
    non_trainable = {}
    
    import re

    for model in models:

        pattern = r'[^/]+$'

        # print("1")
        for var in model.trainable_variables:
            # Extract last part after the last '/'
            match = re.search(pattern, var.name)
            if match:
                matched_name = match.group()
                # print("2")
                get_result = trainable.get(matched_name)
                # print("type(get_result) = ", type(get_result))
                # print("get_result = ", get_result)
                if get_result != None:
                    # print("3")
                    cur_tensor = tf.convert_to_tensor(var)
                    cur_tensor = torch_unsqueeze(cur_tensor, 0)
                    # print("trainable[matched_name] = ", trainable[matched_name])
                    # print("cur_tensor = ", cur_tensor)
                    trainable[matched_name] = torch_cat([tf.convert_to_tensor(trainable[matched_name]), cur_tensor], 0)
                else:
                    # print("4")
                    cur_tensor = tf.convert_to_tensor(var)
                    cur_tensor = torch_unsqueeze(cur_tensor, 0)
                    trainable[matched_name] = cur_tensor
            else:
                if OUTPUT_VARIABLES:
                    print("var.name = ", var.name)
                raise RuntimeError("Network name not recognized!")
        # print("5")
        for var in model.non_trainable_variables:
            # Extract last part after the last '/'
            match = re.search(pattern, var.name)
            if match:
                matched_name = match.group()
                # print("2")
                get_result = non_trainable.get(matched_name)
                # print("type(get_result) = ", type(get_result))
                # print("get_result = ", get_result)
                if get_result != None:
                    # print("3")
                    cur_tensor = tf.convert_to_tensor(var)
                    cur_tensor = torch_unsqueeze(cur_tensor, 0)
                    # print("trainable[matched_name] = ", trainable[matched_name])
                    # print("cur_tensor = ", cur_tensor)
                    non_trainable[matched_name] = torch_cat([tf.convert_to_tensor(non_trainable[matched_name]), cur_tensor], 0)
                else:
                    # print("4")
                    cur_tensor = tf.convert_to_tensor(var)
                    cur_tensor = torch_unsqueeze(cur_tensor, 0)
                    non_trainable[matched_name] = cur_tensor
            else:
                if OUTPUT_VARIABLES:
                    print("var.name = ", var.name)
                raise RuntimeError("Network name not recognized!")


            # print(f"Variable Name: {var.name}")
            # print(f"Variable Shape: {var.shape}")
            # print(f"Variable Type: {type(var)}")
            # print(f"Variable Value (as NumPy array): {var.numpy()}")
            # print("=" * 40)

    #     for var in model.non_trainable_variables:
    #         print(f"Variable Name: {var.name}")
    #         print(f"Variable Shape: {var.shape}")
    #         print(f"Variable Type: {type(var)}")
    #         print(f"Variable Value (as NumPy array): {var.numpy()}")
    #         print("=" * 40)

    # trainable = {model.name: tf.stack([var for var in model.trainable_variables])
    #              for model in models}
    # non_trainable = {model.name: tf.stack([var for var in model.non_trainable_variables])
    #                  for model in models}

    # print("trainable = ", trainable)
    # print("non_trainable = ", non_trainable)

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

    # # Draw values from a normal distribution
    # normal_values = np.random.normal(loc=mean, scale=std, size=variable.shape)

    # # Assign the values to the TensorFlow variable
    # variable.assign(normal_values.astype(np.float32))

    initializer = tf.keras.initializers.RandomNormal(mean=mean, stddev=std)
    initial_value = initializer(shape=variable.shape, dtype=variable.dtype)
    variable.assign(initial_value)




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







def torch_nn_init_xavier_normal_(tensor, gain):
    if not isinstance(tensor, tf.Variable):
        raise ValueError("Input variable must be a tf.Variable.")
    assert len(tensor.shape) == 2, "input tensor must be of shape 2"
    fan_in = tensor.shape[0]
    fan_out = tensor.shape[1]

    std = gain * tf.sqrt( 2 / (fan_in + fan_out) )

    # normal_values = np.random.normal(loc=0, scale=std, size=tensor.shape)

    # # Assign ones to the variable
    # tensor.assign(normal_values.astype(np.float32))

    # Use TensorFlow random number generator
    normal_values = tf.random.normal(
        shape=tensor.shape,
        mean=0.0,
        stddev=std,
        dtype=tensor.dtype  # Automatically match the variable's data type (e.g., float32)
    )

    # Directly assign the generated TensorFlow tensor
    tensor.assign(normal_values)





# def torch_nn_init_trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
def torch_nn_init_trunc_normal_(input_tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, max_tries=1000):
    """
    PyTorch: `torch.nn.init.trunc_normal_`。
    
    Rejection Sampling
    """
    if not isinstance(input_tensor, tf.Variable):
        raise ValueError("Input variable must be a tf.Variable.")

    shape = input_tensor.shape

    a = (a - mean) / std  # Normalize truncation boundaries
    b = (b - mean) / std

    tensor = tf.TensorArray(dtype=tf.float32, size=tf.reduce_prod(shape))
    idx = 0
    num_samples = tf.reduce_prod(shape)

    while idx < num_samples:

        sample = tf.random.normal((num_samples,), mean=mean, stddev=std)

        mask = (sample >= a) & (sample <= b)
        valid_samples = tf.boolean_mask(sample, mask)

        num_valid = tf.shape(valid_samples)[0]
        num_needed = num_samples - idx
        if num_valid > num_needed:
            valid_samples = valid_samples[:num_needed]

        for i in range(tf.shape(valid_samples)[0]):
            tensor = tensor.write(idx, valid_samples[i])
            idx += 1
            if idx >= num_samples:
                break

    tensor_stack = tensor.stack()

    result =  tf.reshape(tensor_stack, shape)

    input_tensor.assign(result)





















import math
def pytorch_weight_initializer(shape, in_features, dtype=None):
    limit = math.sqrt(1.0 / in_features)
    return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype or tf.float32)

def pytorch_bias_initializer(shape, in_features, dtype=None):
    limit = math.sqrt(1.0 / in_features)
    return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype or tf.float32)







class nn_Tanh(tf.keras.layers.Layer):
    def __init__(self, name = "nn_Tanh", **kwargs):
        super(nn_Tanh, self).__init__(name=name, **kwargs)

    def call(self, x):
        return tf.math.tanh(x)


    def get_config(self):
        config = super(nn_Tanh, self).get_config()  # Call the parent layer's get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_Tanh: from_config()")
        
        result = cls(**config)
        return result




class nn_Identity(tf.keras.layers.Layer):
    def __init__(self, name = "nn_Identity", **kwargs):
        super(nn_Identity, self).__init__(name = name, **kwargs)

    def call(self, x):
        # print("nn_Identity: call()")
        return tf.identity(x)



    def get_config(self):
        config = super(nn_Identity, self).get_config()  # Call the parent layer's get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_Identity: from_config()")
        result = cls(**config)
        return result




class nn_Softplus(tf.keras.layers.Layer):
    def __init__(self, name = "nn_Softplus", **kwargs):
        super(nn_Softplus, self).__init__(name = name, **kwargs)

    def call(self, x):
        return tf.math.softplus(x)



    def get_config(self):
        config = super(nn_Softplus, self).get_config()  # Call the parent layer's get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_Softplus: from_config()")
        result = cls(**config)
        return result





class nn_Mish(tf.keras.layers.Layer):
    def __init__(self, name = "nn_Mish", **kwargs):
        super(nn_Mish, self).__init__(name=name, **kwargs)

    def call(self, x):
        # print("nn_Mish.call()")
        result = tf.clip_by_value(x, float('-inf'), 20)

        beta = 1
        result = beta * result

        result = tf.math.softplus(result)

        return x * tf.math.tanh(result)

    
    def get_config(self):
        config = super(nn_Mish, self).get_config()  # Call the parent layer's get_config()
        # config.update({
        #     "inplace": self.inplace,
        #     "relu": tf.keras.layers.serialize(self.relu),
        # })
        return config
    
    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_Mish: from_config()")
        # relu = tf.keras.layers.deserialize(config.pop("relu"))
        result = cls(**config)
        # result.relu=relu
        return result



# Define TensorFlow nn.ELU wrapper
class nn_ELU(tf.keras.layers.Layer):
    def __init__(self, alpha=1.0, name = "nn_ELU", elu = None, **kwargs):
        self.alpha = alpha
        super(nn_ELU, self).__init__(name=name, **kwargs)
        if elu == None:
            self.elu = tf.keras.layers.ELU(alpha=alpha)
        else:
            self.elu = elu
    def call(self, x):
        return self.elu(x)


    def get_config(self):
        config = super(nn_Mish, self).get_config()  # Call the parent layer's get_config()
        config.update({
            "alpha": self.alpha,
            "elu": tf.keras.layers.serialize(self.elu),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_ELU: from_config()")
        elu = tf.keras.layers.deserialize(config.pop("elu"))
        result = cls(elu=elu, **config)
        return result




# # Define TensorFlow nn.GELU wrapper
# class nn_GELU(tf.keras.layers.Layer):
#     def __init__(self):
#         super(nn_GELU, self).__init__()
#         self.gelu = 
#         # tf.nn.gelu()
#         # tf.keras.layers.GELU()

#     def call(self, x):
#         return self.gelu(x)



class nn_GELU(tf.keras.layers.Layer):
    def __init__(self, name = "nn_GELU", **kwargs):
        super(nn_GELU, self).__init__(name=name,**kwargs)

    def call(self, inputs):
        # GELU activation function formula
        return tf.nn.gelu(inputs)


    def get_config(self):
        config = super(nn_GELU, self).get_config()  # Call the parent layer's get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_GELU: from_config()")
        result = cls(**config)
        return result



# Define TensorFlow nn.ReLU wrapper
class nn_ReLU(tf.keras.layers.Layer):
    def __init__(self, inplace=False, name = "nn_ReLU", relu = None, **kwargs):


        super(nn_ReLU, self).__init__(name=name, **kwargs)

        if relu == None:
            self.relu = tf.keras.layers.ReLU()
        else:
            self.relu = relu

        self.inplace = inplace


    def call(self, x):
        if self.inplace:
            x = self.relu(x)
            return x
        else:
            return self.relu(x)


    def get_config(self):
        config = super(nn_ReLU, self).get_config()  # Call the parent layer's get_config()
        config.update({
            "inplace": self.inplace,
            "relu": tf.keras.layers.serialize(self.relu),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_ReLU: from_config()")
        relu = tf.keras.layers.deserialize(config.pop("relu"))
        result = cls(**config)
        result.relu=relu
        return result








class nn_Dropout(tf.keras.layers.Layer):
    def __init__(self, p=0.5, name = "nn_Dropout", model = None, **kwargs):
        super(nn_Dropout, self).__init__(name=name, **kwargs)
        self.p = p
        if model == None:
            self.model = tf.keras.layers.Dropout(rate = p)
        else:
            self.model = model

    # torch.nn.Dropout(p=0.5, inplace=False)
    # tf.keras.layers.Dropout(
    #     rate, noise_shape=None, seed=None, **kwargs
    # )

    def call(self, net_params, training=False):
        return self.model(net_params, training=training)



    def get_config(self):
        config = super(nn_Dropout, self).get_config()  # Call the parent layer's get_config()
        config.update({
            "p": self.p,
            "model": tf.keras.layers.serialize(self.model),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_Dropout: from_config()")

        model = tf.keras.layers.deserialize(config.pop("model"))
        result = cls(model=model, **config)
        return result












class nn_Linear(tf.keras.layers.Layer):
    """
    torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    tf.keras.layers.Dense(
        units,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank=None,
        **kwargs
    )
    """
    def __init__(self, in_features, out_features, 
                #  bias=True, 
                 device=None, dtype=None, 
                #  name="nn_Linear", 
                 name_Dense = None, model=None, **kwargs):
        

        # super(nn_Linear, self).__init__(name=name, **kwargs)
        super(nn_Linear, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        # self.bias = bias
        self.device = device
        # self.dtype=dtype



        if model == None:

            import math
            # # PyTorch-style initialization for weights
            # def pytorch_weight_initializer(shape, dtype=None):
            #     limit = math.sqrt(1.0 / in_features)
            #     return np.random.uniform(-limit, limit, size=shape).astype(np.float32)

            # # PyTorch-style initialization for bias
            # def pytorch_bias_initializer(shape, dtype=None):
            #     # if not bias:
            #     #     return None
            #     limit = math.sqrt(1.0 / in_features)
            #     return np.random.uniform(-limit, limit, size=shape).astype(np.float32)

            # def pytorch_weight_initializer(shape, dtype=None):
            #     limit = math.sqrt(1.0 / in_features)
            #     return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype or tf.float32)

            # def pytorch_bias_initializer(shape, dtype=None):
            #     limit = math.sqrt(1.0 / in_features)
            #     return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype or tf.float32)



            self.model = tf.keras.layers.Dense(
                out_features,
                activation=None,
                use_bias=True,
                kernel_initializer = tf.keras.initializers.Constant(
                    pytorch_weight_initializer((in_features, out_features), in_features)),
                # 'glorot_uniform',
                bias_initializer = tf.keras.initializers.Constant(
                    pytorch_bias_initializer((out_features,), in_features)),
                    # 'zeros',
                dtype=dtype,
                name = name_Dense
                # kernel_regularizer=None,
                # bias_regularizer=None,
                # activity_regularizer=None,
                # kernel_constraint=None,
                # bias_constraint=None,
                # ,**kwargs
            )

            
        else:
            self.model = model
            # print("self.model.trainable_variables = ", self.model.trainable_variables)
            # print("self.model.non_trainable_variables = ", self.model.non_trainable_variables)






        # print("nn_Linear: self.model = ", self.model)

        if OUTPUT_VARIABLES:
            print("nn_Linear: name_Dense = ", name_Dense)
            print("nn_Linear: self.model.name = ", self.model.name)

    def get_config(self):
        # Get the configuration of the layer and return it as a dictionary
        config = super(nn_Linear, self).get_config()  # Call the parent layer's get_config()
        config.update({
            "in_features": self.in_features,
            "out_features": self.out_features,
            # "bias": self.bias,
            # "device": self.device,
            # "dtype": self.dtype,
            "dense_layer": tf.keras.layers.serialize(self.model),
        })
        
        if OUTPUT_VARIABLES:
            print("nn_Linear.config() = ", config)

        return config
    
    @classmethod
    def from_config(cls, config):

        if OUTPUT_FUNCTION_HEADER:
            print("nn_Linear: from_config()")

        model = tf.keras.layers.deserialize(config.pop("dense_layer"))
        result = cls(model = model, **config)
        # result = cls(**config)
        return result


    def call(self, x):

        # print("self.model.kernel = ", self.model.kernel)
        # print("self.model.bias = ", self.model.bias)
        # print("x = ", x)

        result = self.model(x)

        # print("nn_Linear.call() result = ", result)

        # print("nn_Linear.call() result.shape = ", result.shape)

        if OUTPUT_VARIABLES and DEBUG and self.model.built:
            # print("nn_Linear.call() self.kernel = ", self.model.kernel)
            # print("nn_Linear.call() self.bias = ", self.model.bias)
        #     # print("nn_Linear.call() self.kernel = ", self.model.kernel.numpy())  # output kernel value
        #     # print("nn_Linear.call() self.bias = ", self.model.bias.numpy())      # output bias value

            weights = self.model.kernel
            bias = self.model.bias

        #     # print("weights.shape = ", weights.shape)
        #     # print("bias.shape = ", bias.shape)
        #     # print("x.shape = ", x.shape)

            result1 = tf.matmul(x, weights) + bias  # broadcast addition

        #     # result1 = weights * x.numpy() + bias

            print("nn_Linear.call() result1 = ", result1)

        #     assert np.allclose(result1.numpy(), result.numpy())

        return result

    
    # def build(self, input_shape):
    #     self.model.build(input_shape = input_shape)


    @property
    def trainable_variables(self):
        # Return the trainable variables of the inner Dense layer
        return self.model.trainable_variables

    @property
    def non_trainable_variables(self):
        # Return the non-trainable variables of the inner Dense layer
        return self.model.non_trainable_variables

    @property
    def kernel(self):
        return self.model.kernel

    @kernel.setter
    def kernel(self, value):
        self.model.kernel.assign(value)
    
    @property
    def bias(self):
        return self.model.bias

    @bias.setter
    def bias(self, value):
        self.model.bias.assign(value)


    # def __getattr__(self, name):
    #     if hasattr(self.model, name):
    #         return getattr(self.model, name)
    #     else:
    #         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # def __getattr__(self, name):
    #     # First check the current object's attributes to avoid infinite recursion
    #     if name in self.__dict__:
    #         return self.__dict__[name]
    #     # Then check the model's attributes
    #     if hasattr(self.model, name):
    #         return getattr(self.model, name)
    #     # Raise AttributeError if the attribute is not found
    #     raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    # def __getattr__(self, name):
    #     # Avoid recursive calls: directly access __dict__ or object.__getattribute__
    #     try:
    #         return object.__getattribute__(self, name)
    #     except AttributeError:
    #         pass

    #     # Check if the attribute exists in the model
    #     model = object.__getattribute__(self, "model")

    #     if hasattr(model, name):
    #         return getattr(model, name)

    #     # Raise exception if the attribute is still not found
    #     raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")















# class nn_Parameter(tf.keras.layers.Layer):
#     def __init__(self, data=None, requires_grad=True):
#         super().__init__(trainable=requires_grad)  # Important! Ensure trainable is properly initialized
#         if data is None:
#             raise ValueError("data cannot be None. Please provide a tensor value.")
#         if requires_grad:
#             raise ValueError("requires grad version is not implemented right now.")
#             # self.data = tf.Variable(data, trainable=requires_grad, name="nn_parameter")
#         else:
#             self.data =  tf.convert_to_tensor(data)

#     def __tf_tensor__(self):
#         return self.data
    
#     def __array__(self):
#         return self.data.numpy()


#     def numpy(self):
#         return self.data.numpy()
    



def nn_Parameter(data=None, requires_grad=True):
    if data is None:
        raise ValueError("data cannot be None. Please provide a tensor value.")
    # if requires_grad:
    return tf.Variable(data, trainable=requires_grad, name="nn_parameter")
    # else:
    #     return tf.convert_to_tensor(data)



def save_tf_Variable(tensor, save_tensor_name):
    base_path = "/ssddata/qtguo/GENERAL_DATA/"
    params_dict = {}
    result = tensor.numpy()
    params_dict[save_tensor_name] = result

    print("save_tf_Variable: type(tensor) = ", type(tensor))

    if isinstance(tensor, tf.Variable):
        params_dict["trainable"] = tensor.trainable
    elif isinstance(tensor, tf.Tensor):
        params_dict["trainable"] = False
    else:
        raise RuntimeError("save_tf_Variable tensor type wrong")

    import pickle

    pkl_file_path = base_path + save_tensor_name + '.pkl'
    
    print("pkl_file_path = ", pkl_file_path)

    with open(pkl_file_path, 'wb') as f:
        pickle.dump(params_dict, f)






def load_tf_Variable(load_tensor_name):    
    base_path = "/ssddata/qtguo/GENERAL_DATA/"
    pkl_file_path = base_path + load_tensor_name + '.pkl'

    print("pkl_file_path = ", pkl_file_path)

    import pickle

    with open(pkl_file_path, 'rb') as file:
        params_dict = pickle.load(file)

    trainable = params_dict["trainable"]
    result = tf.Variable( params_dict[load_tensor_name] , trainable = trainable)

    return result











# Define TensorFlow nn.Sequential wrapper
class nn_Sequential(tf.keras.layers.Layer):
    def __init__(self, *args, 
                #  name = "nn_Sequential", 
                 model_list = None, 
                 model = None, 
                 **kwargs):
        super(nn_Sequential, self).__init__(
            # name=name, 
            **kwargs)

        if OUTPUT_FUNCTION_HEADER:
            print("nn_Sequential: __init__()")

        if OUTPUT_VARIABLES:
            print("len(args) = ", len(args))
            print("args = ", args)
            print("model_list = ", model_list)
            print("model = ", model)
            print("**kwargs = ", kwargs)


        if model == None:
            self.model_list = []

            if isinstance(args[0], (tuple, list)):
                for module in args[0]:
                    self.model_list.append(module)
            elif isinstance(args[0], (dict, OrderedDict)):
                if OUTPUT_VARIABLES:
                    print("OrderedDict")
                for name, module in args[0].items():
                    if OUTPUT_VARIABLES:
                        print("name = ", name)
                        print("module = ", module)

                    self.model_list.append(module)
            else:
                for module in args:
                    self.model_list.append(module)

            if OUTPUT_VARIABLES:
                print("branch1")
                print("self.model_list = ")

            for i in range(len(self.model_list)):
                if isinstance(self.model_list[i], nn_Sequential):
                    self.model_list[i] = tf.keras.Sequential(self.model_list[i].model_list)

            self.model = tf.keras.Sequential(self.model_list)
        else:
            self.model = model

    
        # if model_list != None:

        # else:
        #     self.model_list = model_list


        # tf.keras.Sequential(
        #     layers=None, trainable=True, name=None
        # )

    def call(self, x):
        output = x
        # if self.model_list:
        #     for module in self.model_list:
        #         print("module = ", module)
        #         output = module(output)
        output = self.model(output)
        
        return output
    
    def __getitem__(self, id):
        # print("getitem: len(self.model_list) = ", len(self.model_list))
        # return self.model_list[id]
        return self.model.layers[id]

    def __iter__(self):
        # return iter(self.model_list)\
        # print("iter: self.model.layers = ", self.model.layers)
        return iter(self.model.layers)

    def __len__(self):
            nn_Sequential_len = len(self.model.layers)
            # print("nn_Sequential_len = ", nn_Sequential_len)
            return nn_Sequential_len
    

    def get_config(self):
        # Get the configuration of all layers in the model_list
        config = super(nn_Sequential, self).get_config()  # Call the parent class get_config()
        
        # # Create a list of layer configurations
        # layer_configs = []


        # for layer in self.model_list:
        #     print("layer = ", layer)
        #     layer_configs.append( layer.get_config() )
        
        # Add the list of layer configurations to the config dictionary
        config.update({
            'model': tf.keras.layers.serialize(self.model)
            # layer_configs
        })

        if OUTPUT_VARIABLES:
            print("nn_Sequential: config = ", config)

        return config

    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_Sequential: from_config()")

        if OUTPUT_VARIABLES:
            print("config = ", config)
        

        from model.diffusion.mlp_diffusion import DiffusionMLP
        from model.diffusion.diffusion import DiffusionModel
        from model.common.mlp import MLP, ResidualMLP
        from model.diffusion.modules import SinusoidalPosEmb
        from model.common.modules import SpatialEmb, RandomShiftsAug
        # from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish

        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
            'DiffusionMLP': DiffusionMLP,
            # 'VPGDiffusion': VPGDiffusion,
            'SinusoidalPosEmb': SinusoidalPosEmb,   
            'MLP': MLP,                            # Custom MLP (Multi-Layer Perceptron) layer
            'ResidualMLP': ResidualMLP,            # Custom ResidualMLP layer
            'nn_Identity': nn_Identity,
            'nn_Sequential': nn_Sequential,        # Custom Sequential class
            'nn_Linear': nn_Linear,
            'nn_LayerNorm': nn_LayerNorm,
            'nn_Dropout': nn_Dropout,
            'nn_ReLU': nn_ReLU,
            'nn_Mish': nn_Mish,
            'SpatialEmb': SpatialEmb,
            'RandomShiftsAug': RandomShiftsAug,
         }
        # Register your custom class with Keras
        get_custom_objects().update(cur_dict)

        # print('get_custom_objects() = ', get_custom_objects())

        # print("Custom objects:", get_custom_objects())
        # assert 'SinusoidalPosEmb' in get_custom_objects()

        # model_list = config.pop("model_list")

        # models = []
        # for model in model_list:
        #     # print("model = ", model)
        #     name = model["name"]
        #     if name in cur_dict:
        #         models.append( cur_dict[name].from_config(model) )
        #     else:
        #         models.append( tf.keras.layers.deserialize( model ,  custom_objects=get_custom_objects() ) )

        model = tf.keras.layers.deserialize( config.pop("model") ,  custom_objects=get_custom_objects() )
        
        return cls(model = model, **config)



class nn_ModuleList(tf.keras.layers.Layer):
    def __init__(self, modules=None, serialized_modules = None, name="nn_ModuleList", **kwargs):
        super(nn_ModuleList, self).__init__(name=name, **kwargs)
        self.modules = []
        if serialized_modules is None:
            if modules is not None:
                for module in modules:
                    self.append(module)
        else:
            self.modules = serialized_modules



    # def get_config(self):
    #     from tensorflow.python.keras.utils import generic_utils
    #     import copy
    #     layer_configs = []
    #     for layer in self.modules:
    #         # `super().layers` include the InputLayer if available (it is filtered out
    #         # of `self.layers`). Note that `self._self_tracked_trackables` is managed
    #         # by the tracking infrastructure and should not be used.
    #         layer_configs.append(generic_utils.serialize_keras_object(layer))
    #     config = {
    #         'name': self.name,
    #         'layers': copy.deepcopy(layer_configs)
    #     }
    #     if not self._is_graph_network and self._build_input_shape is not None:
    #         config['build_input_shape'] = self._build_input_shape
    #     return config


    # @classmethod
    # def from_config(cls, config, custom_objects=None):
    #     if 'name' in config:
    #         name = config['name']
    #         build_input_shape = config.get('build_input_shape')
    #         layer_configs = config['layers']
    #     else:
    #         name = None
    #         build_input_shape = None
    #         layer_configs = config
    #         model = cls(name=name)

    #     from tensorflow.python.keras import layers as layer_module

    #     for layer_config in layer_configs:
    #         layer = layer_module.deserialize(layer_config,
    #                                         custom_objects=custom_objects)
    #         model.add(layer)
    #     if (not model.inputs and build_input_shape and
    #         isinstance(build_input_shape, (tuple, list))):
    #         model.build(build_input_shape)
    #     return model




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


    def get_config(self):

        if OUTPUT_FUNCTION_HEADER:
            print("nn_ModuleList: get_config()")

        # Get the configuration of all layers in the modules list
        config = super(nn_ModuleList, self).get_config()  # Call the parent class get_config()

        # Create a list of module configurations
        module_configs = [module.get_config() for module in self.modules]
        
        # Add the list of module configurations to the config dictionary
        config.update({
            'modules': module_configs
        })
        return config



    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_ModuleList: from_config()")

        model_list = config.pop("modules")

        from model.diffusion.mlp_diffusion import DiffusionMLP
        from model.diffusion.diffusion import DiffusionModel
        from model.common.mlp import MLP, ResidualMLP, TwoLayerPreActivationResNetLinear
        from model.diffusion.modules import SinusoidalPosEmb
        from model.common.modules import SpatialEmb, RandomShiftsAug
        # from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish, nn_Identity

        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
            'DiffusionMLP': DiffusionMLP,
            # 'VPGDiffusion': VPGDiffusion,
            'SinusoidalPosEmb': SinusoidalPosEmb,   
            'MLP': MLP,                            # Custom MLP (Multi-Layer Perceptron) layer
            'ResidualMLP': ResidualMLP,            # Custom ResidualMLP layer
            'nn_Sequential': nn_Sequential,        # Custom Sequential class
            "nn_Identity": nn_Identity,
            'nn_Linear': nn_Linear,
            'nn_LayerNorm': nn_LayerNorm,
            'nn_Dropout': nn_Dropout,
            'nn_ReLU': nn_ReLU,
            'nn_Mish': nn_Mish,
            'SpatialEmb': SpatialEmb,
            'RandomShiftsAug': RandomShiftsAug,
            "TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
         }
        # Register your custom class with Keras
        get_custom_objects().update(cur_dict)

        # print('get_custom_objects() = ', get_custom_objects())


        modules = []

        if OUTPUT_VARIABLES:
            print("type(model_list) = ", type(model_list))
            print("model_list = ", model_list)



        for model in model_list:
            if OUTPUT_VARIABLES:
                print("model = ", model)
            name = model["name"]
            if OUTPUT_VARIABLES:
                print("name = ", name)
            if name in cur_dict:
                modules.append( cur_dict[name].from_config(model) )
            else:
                if OUTPUT_VARIABLES:
                    print("nn_ModuleList: name = ", name)
                modules.append( tf.keras.layers.deserialize( model ,  custom_objects=get_custom_objects() ) )



        return cls(serialized_modules = modules, **config)






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
                 scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None, name="nn_Embedding", **kwargs):
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
        super(nn_Embedding, self).__init__(dtype=dtype, name=name, **kwargs)
        
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


    def get_config(self):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_Embedding: from_config()")

        # Get the configuration of the embedding layer
        config = super(nn_Embedding, self).get_config()  # Call parent class get_config()

        # Add custom arguments to config
        config.update({
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'padding_idx': self.padding_idx,
            'max_norm': self.max_norm,
            'norm_type': self.norm_type,
            'scale_grad_by_freq': self.scale_grad_by_freq,
            'sparse': False,  # Sparse is not used in TensorFlow
            '_freeze': self._freeze,
            # Include _weight if it was initialized with custom weights
            '_weight': self.embeddings.numpy() if hasattr(self.embeddings, 'numpy') else None
        })
        return config


    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_Embedding: from_config()")
        # result = cls(**config)
        result = super(nn_Embedding, cls).from_config(config)
        return result


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





























from tensorflow.keras.saving import register_keras_serializable
@register_keras_serializable(package="Custom")
class nn_LayerNorm(tf.keras.layers.Layer):
    # torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)
    def __init__(self, normalized_shape, eps=1e-5, name="nn_LayerNorm", gamma = None, beta = None, **kwargs):
        """
        A wrapper for PyTorch's nn.LayerNorm in TensorFlow.
        Args:
            normalized_shape (int or tuple): Input shape for layer normalization.
            epsilon (float): A small value to add to the denominator for numerical stability.
        """

        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        super(nn_LayerNorm, self).__init__(name=name, **kwargs)
        self.normalized_shape = normalized_shape
        self.epsilon = eps

        # Define trainable parameters gamma (scale) and beta (offset)

        if gamma == None:
            self.gamma = self.add_weight(
                name="gamma",
                shape=self.normalized_shape,
                initializer="ones",
                trainable=True
            )
        else:
            self.gamma = gamma

        if beta == None:
            self.beta = self.add_weight(
                name="beta",
                shape=self.normalized_shape,
                initializer="zeros",
                trainable=True
            )
        else:
            self.beta = beta

    def get_config(self):
        """
        Returns the configuration of the LayerNorm layer.
        This method is used to save and restore the layer's state.
        """
        config = super(nn_LayerNorm, self).get_config()  # Get the base configuration
        # Ensure that `gamma` and `beta` configuration information is added to the returned config
        config.update({
            'normalized_shape': self.normalized_shape,
            'eps': self.epsilon,
            # 'gamma': self.gamma,
            # 'beta': self.beta
        })

        if OUTPUT_VARIABLES:
            print("nn_LayerNorm.get_config() = ", config)

        return config

    @classmethod
    def from_config(cls, config):
        """
        Returns an instance of the custom layer from its configuration.
        """
        result = super(nn_LayerNorm, cls).from_config(config)
        return result
    
    def call(self, x):
        """
        Forward pass for LayerNorm.

        Args:
            x (tf.Tensor): Input tensor to normalize.

        Returns:
            tf.Tensor: The normalized tensor.
        """
        # print("type(normalized_shape) = ", type(self.normalized_shape))
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



    # def get_config(self):
    #     """
    #     Returns the configuration of the LayerNorm layer.
    #     This method is used to save and restore the layer's state.
    #     """
    #     print("nn_LayerNorm: from_config()")

    #     config = super(nn_LayerNorm, self).get_config()  # Call the parent class get_config

    #     # Add custom arguments to the config
    #     config.update({
    #         'normalized_shape': self.normalized_shape,
    #         'eps': self.epsilon,
    #         # 'gamma': self.gamma,
    #         # 'beta': self.beta
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     """
    #     Returns an instance of the custom layer from its configuration.
    #     """

    #     # Directly call the parent class's `from_config` method.
    #     # TensorFlow will automatically handle the restoration of variables (e.g., gamma and beta).

    #     result = super(nn_LayerNorm, cls).from_config(config)
    #     return result


    # @classmethod
    # def from_config(cls, config):
    #     print("nn_LayerNorm: from_config()")
    #     result = cls(**config)
    #     return result






class nn_MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="nn_MultiheadAttention", **kwargs):
        
        if OUTPUT_FUNCTION_HEADER:
            print("called nn_MultiheadAttention __init__()")

        super(nn_MultiheadAttention, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        # Linear transformations for Query, Key, and Value
        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)

        # Output transformation
        self.output_dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into multiple heads"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """Compute scaled dot-product attention"""
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

        # Apply linear transformation to Q, K, V and split into multiple heads
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Scaled dot-product attention
        output, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)

        # Concatenate the multiple heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        attention_weights = tf.reduce_mean(attention_weights, axis=1)  # take the average of all heads

        # Output transformation
        output = self.output_dense(output)
        return output, attention_weights


    def get_config(self):
        """
        Returns the configuration of the MultiheadAttention layer.
        This method is used to save and restore the layer's state.
        """
        print("nn_MultiheadAttention: get_config()")

        config = super(nn_MultiheadAttention, self).get_config()  # Call the parent class get_config

        # Add custom arguments to the config
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model
        })
        return config









@register_keras_serializable(package="Custom")
class nn_Conv1d(tf.keras.layers.Layer):
    """
    A wrapper for PyTorch's nn.Conv1d in TensorFlow.
    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1.
        padding (int): Padding added to both sides of the input. Default: 0.
        dilation (int): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input to output. Default: 1.
        bias (bool): If True, adds a learnable bias to the output. Default: True.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, name="nn_Conv1d", **kwargs):
        super(nn_Conv1d, self).__init__(name=name, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias


        assert (kernel_size - 1) // 2 == padding, "padding mode is 'same'"

        # Ensure in_channels is divisible by groups
        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")

        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            # padding='valid',  # handle the padding manually
            padding='same',
            dilation_rate=self.dilation,
            groups=self.groups,
            use_bias=self.bias,
            kernel_initializer = tf.keras.initializers.Constant(
                pytorch_weight_initializer((kernel_size, in_channels, out_channels), in_channels * kernel_size / groups )),
            bias_initializer = tf.keras.initializers.Constant(
                pytorch_bias_initializer((out_channels,), in_channels * kernel_size / groups ))
        )        

                    
    def get_config(self):
        """
        Returns the configuration of the Conv1d layer.
        This method is used to save and restore the layer's state.
        """
        config = super(nn_Conv1d, self).get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Returns an instance of the custom layer from its configuration.
        """
        return cls(**config)

    # def build(self, input_shape):
    #     """
    #     Build the layer and initialize the weights.
    #     """
    #     super(nn_Conv1d, self).build(input_shape)

    # def build(self, input_shape):
    #     # Explicitly build the Conv1D layer
    #     self.conv1d.build(input_shape)
    #     super(nn_Conv1d, self).build(input_shape)


    def call(self, x):
        """
        Forward pass for Conv1d.

        Args:
            x (tf.Tensor): Input tensor to convolve.

        Returns:
            tf.Tensor: The convolved tensor.
        """

        # print("nn_Conv1d: self.in_channels = ", self.in_channels)

        # print("nn_Conv1d: self.out_channels = ", self.out_channels)

        # print("nn_Conv1d: self.kernel_size = ", self.kernel_size)

        # print("nn_Conv1d: self.stride = ", self.stride)

        # print("nn_Conv1d: x.shape = ", x.shape)

        # print("nn_Conv1d: self.padding = ", self.padding)
        # print("nn_Conv1d: self.dilation = ", self.dilation)
        # print("nn_Conv1d: self.groups = ", self.groups)
        # print("nn_Conv1d: self.bias = ", self.bias)


        torch_to_tf_input = tf.transpose(x, perm = (0, 2, 1) )
        
        # # Apply padding manually
        # if self.padding > 0:
        #     x = tf.pad(torch_to_tf_input, [[0, 0], [self.padding, self.padding], [0, 0]])

        # result = self.conv1d(x)

        result = self.conv1d(torch_to_tf_input)

        result = tf.transpose(result, perm = (0, 2, 1) )

        return result



    def assign_torch_weights(self, numpy_torch_weights):
        torch_weight = numpy_torch_weights.transpose(2, 1, 0)  
        self.conv1d.kernel.assign(torch_weight)











class nn_Conv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups = 1, conv2d = None, **kwargs):
        super(nn_Conv2d, self).__init__()
        
        # Parse parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        # self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.stride = (stride, stride) if isinstance(stride, int) else stride

        self.bias = bias

        # TensorFlow's Conv2D requires NHWC format

        if conv2d:
            self.conv2d = conv2d
        else:
            self.conv2d = tf.keras.layers.Conv2D(
                filters=self.out_channels,
                kernel_size=self.kernel_size,
                strides=self.stride,
                padding="valid",  # PyTorch padding=0 corresponds to TensorFlow "valid"
                use_bias=self.bias,
                # kernel_initializer=tf.keras.initializers.HeUniform(),  # Use PyTorch Kaiming initialization
                # bias_initializer="zeros" if self.bias else None
                kernel_initializer = tf.keras.initializers.Constant(
                    pytorch_weight_initializer( (self.kernel_size[0], self.kernel_size[1], in_channels // groups, out_channels),\
                                                in_channels * self.kernel_size[0] * self.kernel_size[1] // groups )),
                bias_initializer = tf.keras.initializers.Constant(
                    pytorch_bias_initializer((out_channels,), in_channels * self.kernel_size[0] * self.kernel_size[1] // groups ))
            )


    def call(self, x):
        # PyTorch input (N, C, H, W) -> TensorFlow (N, H, W, C)
        x = tf.transpose(x, [0, 2, 3, 1])

        # have convolution
        x = self.conv2d(x)

        # Convert back to PyTorch format: (N, C, H, W)
        x = tf.transpose(x, [0, 3, 1, 2])
        return x


    def get_config(self):
        """
        Returns the configuration of the Conv1d layer.
        This method is used to save and restore the layer's state.
        """
        config = super(nn_Conv2d, self).get_config()

        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'bias': self.bias,
        })

        config.update({
            "conv2d": tf.keras.layers.serialize(self.conv2d)
        })


        return config

    @classmethod
    def from_config(cls, config):
        """
        Returns an instance of the custom layer from its configuration.
        """


        from model.diffusion.mlp_diffusion import DiffusionMLP
        from model.diffusion.diffusion import DiffusionModel
        from model.common.mlp import MLP, ResidualMLP
        from model.diffusion.modules import SinusoidalPosEmb
        from model.common.modules import SpatialEmb, RandomShiftsAug
        from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish

        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
            'DiffusionMLP': DiffusionMLP,
            # 'VPGDiffusion': VPGDiffusion,
            'SinusoidalPosEmb': SinusoidalPosEmb,   
            'MLP': MLP,                            # Custom MLP (Multi-Layer Perceptron) layer
            'ResidualMLP': ResidualMLP,            # Custom ResidualMLP layer
            'nn_Sequential': nn_Sequential,        # Custom Sequential class
            'nn_Linear': nn_Linear,
            'nn_LayerNorm': nn_LayerNorm,
            'nn_Dropout': nn_Dropout,
            'nn_ReLU': nn_ReLU,
            'nn_Mish': nn_Mish,
            'SpatialEmb': SpatialEmb,
            'RandomShiftsAug': RandomShiftsAug,
         }
        # Register your custom class with Keras
        get_custom_objects().update(cur_dict)

        conv2d = config.pop("conv2d")
        conv2d = tf.keras.layers.deserialize( conv2d,  custom_objects=get_custom_objects() )

        return cls(conv2d = conv2d, **config)



    def assign_torch_weights(self, numpy_torch_weights):
        torch_weights_tf = np.transpose(numpy_torch_weights, (2, 3, 1, 0))  # Convert to TensorFlow shape
        self.conv2d.kernel.assign(torch_weights_tf)















class nn_ConvTranspose1d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, \
                bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        super(nn_ConvTranspose1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.dilation = dilation


        in_features = (out_channels * kernel_size) / groups

        # kernel_initializer = tf.keras.initializers.Constant(
        #     pytorch_weight_initializer((in_channels, out_channels // groups, kernel_size), in_features))
        kernel_initializer = tf.keras.initializers.Constant(
            pytorch_weight_initializer((kernel_size, out_channels // groups, in_channels), in_features))

        bias_initializer = tf.keras.initializers.Constant(
            pytorch_bias_initializer((out_channels,), in_features) )


        self.conv1d_transpose = tf.keras.layers.Conv1DTranspose(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            # padding="valid",  # padding manually
            padding="same",
            use_bias=bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )

    def build(self, input_shape):
        self.conv1d_transpose.build((None, None, self.in_channels))


    def call(self, x):
    
        # L_in = x.shape[2]
        # L_out = (L_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + self.output_padding + 1

        # PyTorch (N, C, L) -> TensorFlow (N, L, C)
        x = tf.transpose(x, [0, 2, 1])

        x = self.conv1d_transpose(x)

        # **Convert back to PyTorch format (N, C, L)**
        x = tf.transpose(x, [0, 2, 1])

        return x


    def assign_torch_weights(self, numpy_torch_weights):
        torch_weights_tf = np.transpose(numpy_torch_weights, (2, 1, 0))  # Convert to TensorFlow shape
        self.conv1d_transpose.kernel.assign(torch_weights_tf)













@register_keras_serializable(package="Custom")
class nn_GroupNorm(tf.keras.layers.Layer):
    """
    A wrapper for PyTorch's nn.GroupNorm in TensorFlow.
    Args:
        num_groups (int): Number of groups to divide the channels into.
        num_channels (int): Number of channels in the input tensor.
        eps (float): A small value to add to the denominator for numerical stability.
        affine (bool): If True, learnable scaling (gamma) and offset (beta) parameters are applied.
    """
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, name="nn_GroupNorm", **kwargs):
        super(nn_GroupNorm, self).__init__(name=name, **kwargs)
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        # Ensure num_channels is divisible by num_groups
        if self.num_channels % self.num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        # Define trainable parameters gamma (scale) and beta (offset) if affine is True
        if self.affine:
            self.gamma = self.add_weight(
                name="gamma",
                shape=(self.num_channels,),
                initializer="ones",
                trainable=True
            )
            self.beta = self.add_weight(
                name="beta",
                shape=(self.num_channels,),
                initializer="zeros",
                trainable=True
            )
        else:
            self.gamma = None
            self.beta = None

    def get_config(self):
        """
        Returns the configuration of the GroupNorm layer.
        This method is used to save and restore the layer's state.
        """
        config = super(nn_GroupNorm, self).get_config()
        config.update({
            'num_groups': self.num_groups,
            'num_channels': self.num_channels,
            'eps': self.eps,
            'affine': self.affine,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Returns an instance of the custom layer from its configuration.
        """
        return cls(**config)

    def call(self, x):
        """
        Forward pass for GroupNorm.

        Args:
            x (tf.Tensor): Input tensor to normalize.

        Returns:
            tf.Tensor: The normalized tensor.
        """
        x = torch_tensor_permute(x, [0, 2, 3, 1])

        print("x.shape = ", x.shape)

        # Reshape the input tensor to group the channels
        batch_size, height, width, channels = tf.unstack(tf.shape(x))
        # batch_size, channels, height, width = tf.unstack(tf.shape(x))

        group_size = channels // self.num_groups

        # print("nn_GroupNorm: self.num_groups = ", self.num_groups)
        # print("nn_GroupNorm: self.num_channels = ", self.num_channels)
        # print("nn_GroupNorm: batch_size = ", batch_size)
        # print("nn_GroupNorm: height = ", height)
        # print("nn_GroupNorm: width = ", width)
        # print("nn_GroupNorm: channels = ", channels)
        # print("nn_GroupNorm: self.num_groups = ", self.num_groups)
        # print("nn_GroupNorm: group_size  = ", group_size )

        x = tf.reshape(x, [batch_size, height, width, self.num_groups, group_size])

        # Compute mean and variance for each group
        mean, variance = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)

        # Normalize the input
        normalized_x = (x - mean) / tf.sqrt(variance + self.eps)

        # Reshape back to the original shape
        normalized_x = tf.reshape(normalized_x, [batch_size, height, width, channels])

        # Apply scaling (gamma) and offset (beta) if affine is True
        if self.affine:
            normalized_x = self.gamma * normalized_x + self.beta

        normalized_x = torch_tensor_permute(normalized_x, [0,3,1,2])
        return normalized_x

















@register_keras_serializable(package="Custom")
class einops_layers_torch_Rearrange(tf.keras.layers.Layer):
    """
    A wrapper for PyTorch's nn.GroupNorm in TensorFlow.
    Args:
        num_groups (int): Number of groups to divide the channels into.
        num_channels (int): Number of channels in the input tensor.
        eps (float): A small value to add to the denominator for numerical stability.
        affine (bool): If True, learnable scaling (gamma) and offset (beta) parameters are applied.
    """
    def __init__(self, dimension_str, name="einops_layers_torch_Rearrange", **kwargs):
        super(einops_layers_torch_Rearrange, self).__init__(name=name, **kwargs)
        self.dimension_str = dimension_str

    def get_config(self):
        """
        Returns the configuration of the einops_layers_torch_Rearrange layer.
        This method is used to save and restore the layer's state.
        """
        config = super(einops_layers_torch_Rearrange, self).get_config()
        config.update({
            'dimension_str': self.dimension_str,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Returns an instance of the custom layer from its configuration.
        """
        return cls(**config)

    def call(self, x):
        from einops.layers.tensorflow import Rearrange
        layer = Rearrange(self.dimension_str)
        return layer(x)

























class nn_TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, 
                #  dropout=0.1, 
                 dropout = 0,
                 activation=nn_ReLU, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False):
        super(nn_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn_MultiheadAttention(d_model, nhead, 
                                            #    dropout=dropout, 
                                            #    batch_first=batch_first
                                               )
        self.multihead_attn = nn_MultiheadAttention(d_model, nhead, 
                                                    # dropout=dropout, 
                                                    # batch_first=batch_first
                                                    )
        
        # Feed-Forward Network (FFN)
        self.linear1 = nn_Linear(d_model, dim_feedforward)
        self.linear2 = nn_Linear(dim_feedforward, d_model)
        self.dropout = nn_Dropout(dropout)

        # Normalization
        self.norm1 = nn_LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn_LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn_LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout
        self.dropout1 = nn_Dropout(dropout)
        self.dropout2 = nn_Dropout(dropout)
        self.dropout3 = nn_Dropout(dropout)

        self.activation = activation
        self.norm_first = norm_first  # Control whether normalization is applied first (Pre-LN vs. Post-LN)


    def forward(self, tgt, memory, 
                tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:            
            tgt: Target sequence (embeddings of the target) (batch, tgt_len, d_model)
            memory: Encoder output (batch, src_len, d_model)
            tgt_mask: Attention mask for the target sequence
            memory_mask: Attention mask for encoder-decoder attention
            tgt_key_padding_mask: Padding mask for the target sequence
            memory_key_padding_mask: Padding mask for the encoder output            
        """

        # 1. Self-Attention (Decoder)
        if self.norm_first:
            tgt2 = self.self_attn(self.norm1(tgt), self.norm1(tgt), self.norm1(tgt),
                                  attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
        else:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
            tgt = self.norm1(tgt + self.dropout1(tgt2))

        # 2. Cross-Attention (Encoder-Decoder)
        if self.norm_first:
            tgt2 = self.multihead_attn(self.norm2(tgt), self.norm2(memory), self.norm2(memory),
                                       attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
        else:
            tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
            tgt = self.norm2(tgt + self.dropout2(tgt2))

        # 3. Feedforward Network (FFN)
        if self.norm_first:
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tgt)))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = self.norm3(tgt + self.dropout3(tgt2))

        return tgt










# class nn_TransformerDecoderLayer(tf.keras.layers.Layer):
#     def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, 
#                  name="nn_TransformerDecoderLayer", **kwargs):

#         if OUTPUT_FUNCTION_HEADER:
#             print("called nn_TransformerDecoderLayer __init__()")

#         super(nn_TransformerDecoderLayer, self).__init__(name=name, **kwargs)
#         # self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
#         # self.cross_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)

#         self.self_attn = nn_MultiheadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
#         self.cross_attn = nn_MultiheadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)


#         self.ffn = nn_Sequential([
#         # tf.keras.Sequential([
#             # tf.keras.layers.Dense(dim_feedforward, activation=activation),
#             nn_Linear(dim_feedforward, activation=activation),
#             tf.keras.layers.Dropout(dropout),
#             # tf.keras.layers.Dense(d_model),
#             nn_Linear(d_model),
#         ])

        
#         # self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         # self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         # self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

#         self.norm1 = nn_LayerNorm(epsilon=1e-6)
#         self.norm2 = nn_LayerNorm(epsilon=1e-6)
#         self.norm3 = nn_LayerNorm(epsilon=1e-6)

#         # self.dropout1 = tf.keras.layers.Dropout(dropout)
#         # self.dropout2 = tf.keras.layers.Dropout(dropout)
#         # self.dropout3 = tf.keras.layers.Dropout(dropout)

#         self.dropout1 = nn_Dropout(dropout)
#         self.dropout2 = nn_Dropout(dropout)
#         self.dropout3 = nn_Dropout(dropout)

#     def call(self, tgt, memory, tgt_mask=None, memory_mask=None, training=None):
#         # Self-attention on target
#         tgt2 = self.self_attn(tgt, tgt, attention_mask=tgt_mask, training=training)
#         tgt = tgt + self.dropout1(tgt2, training=training)
#         tgt = self.norm1(tgt)

#         # Cross-attention between target and memory
#         tgt2 = self.cross_attn(tgt, memory, attention_mask=memory_mask, training=training)
#         tgt = tgt + self.dropout2(tgt2, training=training)
#         tgt = self.norm2(tgt)

#         # Feedforward network
#         tgt2 = self.ffn(tgt, training=training)
#         tgt = tgt + self.dropout3(tgt2, training=training)
#         tgt = self.norm3(tgt)

#         return tgt


class nn_TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, nhead, dim_feedforward, dropout, activation, name="nn_TransformerDecoder", **kwargs):

        if OUTPUT_FUNCTION_HEADER:
            print("called nn_TransformerDecoder __init__()")

        super(nn_TransformerDecoder, self).__init__(name=name, **kwargs)
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
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, name="nn_TransformerEncoderLayer", **kwargs):

        if OUTPUT_FUNCTION_HEADER:
            print("called nn_TransformerEncoderLayer __init__()")

        super(nn_TransformerEncoderLayer, self).__init__(name=name, **kwargs)
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
    def __init__(self, n_layers, d_model, nhead, dim_feedforward, dropout, activation, name="nn_TransformerEncoder", **kwargs):

        if OUTPUT_FUNCTION_HEADER:
            print("called nn_TransformerEncoder __init__()")

        super(nn_TransformerEncoder, self).__init__(name=name, **kwargs)
        self.layers = [
            nn_TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(n_layers)
        ]

    def call(self, x, training):
        for layer in self.layers:
            x = layer(x, training=training)
        return x












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


    # def apply_gradients(self, gradients):
    #     self.optimizer.apply_gradients(zip(gradients, self.params))

    def apply_gradients(self, zipped_gradients):
        self.optimizer.apply_gradients(zipped_gradients)






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

        # # TensorFlow AdamW optimizer
        # self.optimizer = tf.keras.optimizers.experimental.AdamW(
        #     learning_rate=lr, beta_1=betas[0], beta_2=betas[1], epsilon=eps, weight_decay=weight_decay
        # )
        # TensorFlow AdamW optimizer
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr, beta_1=betas[0], beta_2=betas[1], epsilon=eps, weight_decay=weight_decay
        )

    def zero_grad(self):
        """No-op function for compatibility, gradients are reset automatically in TensorFlow."""
        pass

    def step(self, gradients):
        """Apply gradients to parameters."""
        self.optimizer.apply_gradients(zip(gradients, self.params))


    # def apply_gradients(self, gradients):
    #     self.optimizer.apply_gradients(zip(gradients, self.params))

    def apply_gradients(self, zipped_gradients):
        self.optimizer.apply_gradients(zipped_gradients)










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
    This function has an additional `grads` parameter because TensorFlow gradients need to be right after tf.GradientTape.
    Mimicking PyTorch's clip_grad_norm_ function, it clips the gradients of TensorFlow model parameters.
    
    :param grads: List of gradients
    :param max_norm: Maximum norm of the gradients
    :param norm_type: Type of norm, default is L2 norm
    :param error_if_nonfinite: Whether to raise an error if gradients contain non-finite values (e.g., NaN or Inf)
    :return: Clipped gradients
    """
    if norm_type != 2.0:
        raise NotImplementedError("Only L2 norm is currently supported")

    # Compute the norm of all gradients
    grads_finite = [tf.clip_by_value(g, -1e7, 1e7) if g is not None else tf.zeros_like(parameters[i]) for i, g in enumerate(grads)]
    # print("grads_finite = ", grads_finite)

    global_norm = tf.norm(tf.stack([tf.norm(grad) for grad in grads_finite if grad is not None]))

    # print("global_norm = ", global_norm)

    # Clip if the norm exceeds the maximum value
    clip_coef = max_norm / (global_norm + 1e-6)
    # print("clip_coef = ", clip_coef)
    clip_coef_bf = tf.where(global_norm < max_norm, tf.ones_like(clip_coef), clip_coef)
    # print("clip_coef_bf = ", clip_coef_bf)
    clipped_grads = [grad * clip_coef_bf if grad is not None else None for grad in grads]

    # print("clipped_grads = ", clipped_grads)

    # If error_if_nonfinite is True, check for non-finite values
    if error_if_nonfinite:
        for g in grads:
            if g is not None and (tf.reduce_any(tf.is_nan(g)) or tf.reduce_any(tf.is_inf(g))):
                raise ValueError("Gradients contain non-finite values.")

    zip_gradients_params = zip(clipped_grads, parameters)

    optimizer.apply_gradients(zip_gradients_params)
    
    return clipped_grads














# def torch_tensor_requires_grad_(tensor, requires_grad=True):
#     # torch.tensor.requires_grad_
#     tensor.trainable = requires_grad
#     return tensor


def torch_tensor_requires_grad_(tensor, requires_grad=True):
    if requires_grad:
        return tf.Variable(tensor, trainable=True)
    else:
        return tf.Variable(tensor, trainable=False)
    


































# class torch_utils_data_DataLoader:
#     # dataset,          # The dataset
#     # batch_size=32,    # Batch size
#     # shuffle=True,     # Shuffle the data during training
#     # num_workers=4,    # 4 subprocesses for loading data
#     # pin_memory=True,  # Enable when training on GPU
#     # drop_last=False,  # Retain the last incomplete batch
#     # prefetch_factor=2 # Preload 2 batches per worker
#     def __init__(self,
#         dataset, 
#         batch_size=1, 
#         shuffle=False, 
#         # sampler=None,
#         # batch_sampler=None, 
#         num_workers=0, 
#         # collate_fn=None,
#         pin_memory=False, 
#         drop_last=False, 
#         # timeout=0,
#         # worker_init_fn=None, *, 
#         prefetch_factor=2,
#         # persistent_workers=False
#         ):
#         # torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
#         #    batch_sampler=None, num_workers=0, collate_fn=None,
#         #    pin_memory=False, drop_last=False, timeout=0,
#         #    worker_init_fn=None, *, prefetch_factor=2,
#         #    persistent_workers=False)
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.drop_last = drop_last
#         self.prefetch_factor = prefetch_factor

    
#     def 

    
#     pass

class torch_utils_data_DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, 
            num_workers=0, pin_memory=False, drop_last=False, prefetch_factor=2):
        """
        PyTorch-style DataLoader implementation, supports TensorFlow-compatible data loading.
        :param dataset: The dataset, must be a list[dict] or implement __getitem__ and __len__
        :param batch_size: The size of each batch
        :param shuffle: Whether to shuffle the data
        :param drop_last: Whether to drop the last incomplete batch
        :param n_epochs: Total number of training epochs
        :param seed: Random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        # self.n_epochs = n_epochs
        # self.seed = seed
        self.data_size = len(dataset)


    def __iter__(self):
        """
        Iterator method, allows traversal of data using `for batch in dataloader`
        """
        # self.seed
        rng = np.random.default_rng()  # Maintain reproducibility
        indices = list(range(self.data_size))

        # for epoch in range(self.n_epochs):
        if self.shuffle:
            rng.shuffle(indices)  # Shuffle the index order
        
        
        batch_data = {
            "actions": [],
            "states": [],
            "rewards": [],
            "next_states": [],
            "rgb": [],
        }
        batch_count = 0

        for i in indices:
            sample = self.dataset[i]  # Get a sample
            batch_data["actions"].append(sample.get("actions", None))
            batch_data["states"].append(sample.get("states", None))
            batch_data["rewards"].append(sample.get("rewards", None))
            batch_data["next_states"].append(sample.get("next_states", None))
            batch_data["rgb"].append(sample.get("rgb", None))

            batch_count += 1

            if batch_count == self.batch_size:  # yield when batch_size is met
                return_dict = {}
                for key, value in batch_data.items():
                    # print("key = ", key)
                    # print("value = ", value) 
                    # print("value[0] = ", value[0])                   
                    if value[0] is not None:
                        return_dict[key] = tf.convert_to_tensor(value)
                    #     # print("return_dict[key].shape = ", return_dict[key].shape)
                    # else:
                    #     return_dict[key] = None
                yield return_dict
                batch_data = {key: [] for key in batch_data}  # Reinitialize
                batch_count = 0

        if not self.drop_last and batch_count > 0:
            # Handle the last batch (if drop_last=False)
            # yield {key: tf.convert_to_tensor(value) for key, value in batch_data.items()}
                return_dict = {}
                for key, value in batch_data.items():
                    # print("key = ", key)
                    # print("value = ", value)
                    if value[0] is not None:   
                        return_dict[key] = tf.convert_to_tensor(value)
                    # else:
                    #     return_dict[key] = None
                yield return_dict


    def __len__(self):
        """
        Calculate the total number of batches
        """
        if self.drop_last:
            return self.data_size // self.batch_size
        return (self.data_size + self.batch_size - 1) // self.batch_size  # Round up
























def torch_no_grad():
    return tf.GradientTape(watch_accessed_variables=False, persistent=True)













# def torch_tensor_cpu(tensor):
#     return 










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



















class CosineAWR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, first_cycle_steps, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=0, gamma=1.0, last_epoch = -1):
        assert warmup_steps < first_cycle_steps

        super(CosineAWR, self).__init__()
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
        # print("CosineAWR.__call__()")
        # print("tf: epoch = ", epoch)

        # print("tf: type(epoch) = ", type(epoch))
        # print("tf: epoch = ", epoch)

        # print("tf: self.last_epoch = ", self.last_epoch)

        #Because tensorflow automatically set epoch for each epoch，to achieve the same optimizer as the pytorch version, we choose to fix epoch=None manually
        if epoch is not None:
            epoch = int(epoch)  # Force conversion to integer

        # print("tf: 2epoch = ", epoch)
        epoch = None
        # print("tf: 3epoch = ", epoch)

        # print("tf: self.base_lr = ", self.base_lr)

        if epoch is None:
            # print("tf: step: branch1")
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                # print("tf: step: branch1-1")
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            # print("tf: step: branch2")
            if epoch >= self.first_cycle_steps:
                # print("tf: step: branch2-1")
                if self.cycle_mult == 1.0:
                    # print("tf: step: branch2-1-1")
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    # print("tf: step: branch2-1-2")
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

                    # print("tf: self.cycle = ", self.cycle)

                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )

                    # print("tf: self.step_in_cycle = ", self.step_in_cycle)

                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )

                    # print("tf: self.cur_cycle_steps = ", self.cur_cycle_steps)

            else:
                # print("tf: step: branch2-2")
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)


        if self.step_in_cycle == -1:
            # return self.base_lrs
            # print("tf: get_lr: branch1")
            self.lr = self.base_lr
        elif self.step_in_cycle < self.warmup_steps:
            # print("tf: get_lr: branch2")
            self.lr = (self.max_lr - self.base_lr) * self.step_in_cycle / self.warmup_steps + self.base_lr
            # [
                # for base_lr in self.base_lrs
            # ]
        else:
            # print("tf: get_lr: branch3")
            self.lr = self.base_lr + (self.max_lr - self.base_lr) * ( 1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                ) / 2

        # update learning rate
        # tf.keras.backend.set_value(self.optimizer.lr, lr)

        # print("tf: lr = ", self.lr)

        return self.lr








# def torch_nn_functional_grid_sample(image, grid, mode='bilinear', padding_mode="zeros", align_corners=False):


#     def grid_sampler_unnormalize_tf(coord, side, align_corners):
#         if align_corners:
#             return ((coord + 1) / 2) * (side - 1)
#         else:
#             return ((coord + 1) * side - 1) / 2
            
#     def grid_sampler_compute_source_index_tf(coord, size, align_corners):
#         return grid_sampler_unnormalize_tf(coord, size, align_corners)

#     def safe_get_tf(image, n, c, x, y, H, W):
#         value = tf.zeros([1])
#         x = tf.cast(x, tf.int32)  # Ensure x is an integer type
#         y = tf.cast(y, tf.int32)  # Ensure y is an integer type
#         if x >= 0 and x < W and y >= 0 and y < H:
#             value = image[n, c, y, x]
#         return value


#     assert padding_mode == "zeros", "only zeros padding_mode is implemented right now"
#     assert padding_mode == "zeros", "only zeros padding_mode is implemented right now"
#     assert mode == "bilinear", "only bilinear is implemented right now"
#     assert len(image.shape) == 4, "len(input.shape) must be 4"

#     N, C, H_in, W_in = image.shape

#     H_out = grid.shape[1]
#     W_out = grid.shape[2]
    
#     # output_tensor = tf.zeros_like(image)
    
#     output_tensor = np.zeros( [N, C, H_out, W_out] )
#     # np.zeros_like(image)



#     for n in range(N):
#         for w in range(W_out):
#             for h in range(H_out):
#                 # Get corresponding grid x and y
#                 x = grid[n, h, w, 1]
#                 y = grid[n, h, w, 0]
                
#                 # Unnormalize with align_corners condition
#                 ix = grid_sampler_compute_source_index_tf(x, W_in, align_corners)
#                 iy = grid_sampler_compute_source_index_tf(y, H_in, align_corners)
                
#                 x0 = tf.floor(ix)
#                 x1 = x0 + 1

#                 y0 = tf.floor(iy)
#                 y1 = y0 + 1
    
#                 # Get W matrix before I matrix, as I matrix requires Channel information
#                 wa = (x1 - ix) * (y1 - iy)
#                 wb = (x1 - ix) * (iy - y0)
#                 wc = (ix - x0) * (y1 - iy)
#                 wd = (ix - x0) * (iy - y0)
                
#                 # Get values of the image by provided x0, y0, x1, y1 by channel
#                 for c in range(C):
#                     Ia = safe_get_tf(image, n, c, y0, x0, H_in, W_in)
#                     Ib = safe_get_tf(image, n, c, y1, x0, H_in, W_in)
#                     Ic = safe_get_tf(image, n, c, y0, x1, H_in, W_in)
#                     Id = safe_get_tf(image, n, c, y1, x1, H_in, W_in)
#                     out_ch_val = Ia * wa + Ib * wb + Ic * wc + Id * wd

#                     # output_tensor[n, h, w, c] = out_ch_val
#                     # output_tensor[n, h, w, c] = out_ch_val.numpy()
#                     output_tensor[n, c, h, w] = out_ch_val.numpy()
#     output_tensor = tf.convert_to_tensor(output_tensor)
#     return output_tensor






def torch_nn_functional_grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    def safe_gather_nd(tensor, indices, default_value=0.0):

        tensor_len = len(tensor.shape)
        indices_last_len = tf.shape(indices)[-1]
        augment_dim = tensor_len - indices_last_len
        

        # print("tensor = ", tensor)
        # print("indices = ", indices)
        # Get tensor shape
        tensor_shape = tf.shape(tensor)
        max_indices = tensor_shape[:tf.shape(indices)[-1]]  # Calculate maximum index for each dimension

        # print("max_indices = ", max_indices)

        # Check if indices are out of bounds
        is_out_of_bounds = tf.reduce_any(indices < 0, axis=-1) | tf.reduce_any(indices >= max_indices, axis=-1)

        match_dim_is_out_of_bounds = tf.expand_dims(is_out_of_bounds, axis=-1)  # (3, 1)
        
        # print("is_out_of_bounds = ", is_out_of_bounds)
        # print("match_dim_is_out_of_bounds = ", match_dim_is_out_of_bounds)
        # print("tf.zeros_like(indices) = ", tf.zeros_like(indices))
        # print("indices = ", indices)

        # Create valid indices by replacing out-of-bounds indices with (0, 0, ...) to avoid errors
        safe_indices = tf.where(match_dim_is_out_of_bounds, tf.zeros_like(indices), indices)

        # print("safe_indices = ", safe_indices)

        # Get gather_nd result
        gathered_values = tf.gather_nd(tensor, safe_indices)

        # print("gathered_values = ", gathered_values)
        # print("gathered_values = ", tf.transpose(gathered_values, [0, 3, 1, 2]))

        for i in range(augment_dim):
            is_out_of_bounds = tf.expand_dims(is_out_of_bounds, axis=-1)

        # print("is_out_of_bounds = ", is_out_of_bounds)

        # Replace out-of-bounds indices with default_value
        result = tf.where(is_out_of_bounds, tf.fill(tf.shape(gathered_values),  tf.cast(default_value, gathered_values.dtype) ), gathered_values)

        # print("result = ", result)

        return result

    assert mode == "bilinear", "Only bilinear mode is supported."
    assert padding_mode == "zeros", "Only zeros padding_mode is implemented."

    # PyTorch: [N, C, H, W] → TensorFlow: [N, H, W, C]
    # x = tf.transpose(x, [0, 2, 3, 1])
    # N, H_in, W_in, C = x.shape
    N, C, H_in, W_in = image.shape

    H_out, W_out = grid.shape[1:3]

    # Normalize grid [-1, 1] → [0, H-1] or [0, W-1]
    def grid_sampler_unnormalize(coord, size, align_corners):
        # if align_corners:
        #     return 0.5 * ((coord + 1) * (size - 1))
        # else:
        #     return 0.5 * ((coord + 1) * size - 1)
        if align_corners:
            return ((coord + 1) / 2) * (size - 1)
        else:
            return ((coord + 1) * size - 1) / 2
            
    # dim1 = tf.reshape( tf.range(N), [-1, 1, 1, 1] )

    # dim2 = tf.reshape( tf.range(H_in), [1, -1, 1, 1])
    # dim3 = tf.reshape( tf.range(W_in), [1, 1, -1, 1])
    # dim4 = tf.reshape( tf.range(2), [1, 1, 1, -1])

    # batch_idx = tf.reshape(tf.range(N), [-1, 1, 1])  # [N, 1, 1]
    # batch_idx = tf.broadcast_to(batch_idx, [N, y.shape[1], y.shape[2]])  # Expand to [N, H_out, W_out]
    
    # indices_x = tf.stack([dim1, dim3, dim3, dim4], axis=-1)
    
    # xy = tf.gather_nd(image, indices_x)
    


    # ix = grid_sampler_unnormalize(grid[..., 0], W_in, align_corners)
    # iy = grid_sampler_unnormalize(grid[..., 1], H_in, align_corners)
    ix = grid_sampler_unnormalize(grid[..., 1], W_in, align_corners)
    iy = grid_sampler_unnormalize(grid[..., 0], H_in, align_corners)
    
    # print("image.shape = ", image.shape)
    # print("grid.shape = ", grid.shape)
    # print("ix.shape = ", ix.shape)
    # print("iy.shape = ", iy.shape)

    x0 = tf.math.floor(ix)
    x1 = x0 + 1

    y0 = tf.math.floor(iy)
    y1 = y0 + 1

    # print("y0 = ", y0)
    # print("y0.shape = ", y0.shape)

    # print("y1 = ", y1)

    # print("x0.shape = ", x0.shape)
    # print("x1.shape = ", x1.shape)
    # print("y0.shape = ", y0.shape)
    # print("y1.shape = ", y1.shape)

    # # Limit index range (padding_mode="zeros" handles the boundaries)
    # x0 = tf.clip_by_value(x0, 0, W_in - 1)
    # x1 = tf.clip_by_value(x1, 0, W_in - 1)
    # y0 = tf.clip_by_value(y0, 0, H_in - 1)
    # y1 = tf.clip_by_value(y1, 0, H_in - 1)

    # Compute weights
    wa = (x1 - ix) * (y1 - iy)
    wb = (x1 - ix) * (iy - y0)
    wc = (ix - x0) * (y1 - iy)
    wd = (ix - x0) * (iy - y0)


    def gather_nd(image, x, y, N):
        """ Use tf.gather_nd for indexing, ensuring batch_idx matches the shape """
        batch_idx = tf.reshape(tf.range(N), [-1, 1, 1])  # [N, 1, 1]
        batch_idx = tf.broadcast_to(batch_idx, [N, y.shape[1], y.shape[2]])  # Expand to [N, H_out, W_out]
        
        indices = tf.stack([batch_idx, x, y], axis=-1)  # [N, H_out, W_out, 3]
        return safe_gather_nd(image, indices)

    

    image = tf.transpose(image, [0, 2, 3, 1])

    Ia = gather_nd(image, tf.cast(x0, tf.int32), tf.cast(y0, tf.int32), N)
    Ib = gather_nd(image, tf.cast(x0, tf.int32), tf.cast(y1, tf.int32), N)
    Ic = gather_nd(image, tf.cast(x1, tf.int32), tf.cast(y0, tf.int32), N)
    Id = gather_nd(image, tf.cast(x1, tf.int32), tf.cast(y1, tf.int32), N)


    # print("Ia = ", Ia)
    
    Ia = tf.transpose(Ia, [0, 3, 1, 2])
    Ib = tf.transpose(Ib, [0, 3, 1, 2])
    Ic = tf.transpose(Ic, [0, 3, 1, 2])
    Id = tf.transpose(Id, [0, 3, 1, 2])

    # print("Ia = ", Ia)
    # print("Ib = ", Ib)
    # print("Ic = ", Ic)
    # print("Id = ", Id)

    wa = tf.expand_dims(wa, axis=1)  # Change to [N, 1, H_out, W_out]
    wa = tf.broadcast_to(wa, [N, C, H_out, W_out])
    wb = tf.expand_dims(wb, axis=1)  # Change to [N, 1, H_out, W_out]
    wb = tf.broadcast_to(wb, [N, C, H_out, W_out])
    wc = tf.expand_dims(wc, axis=1)  # Change to [N, 1, H_out, W_out]
    wc = tf.broadcast_to(wc, [N, C, H_out, W_out])
    wd = tf.expand_dims(wd, axis=1)  # Change to [N, 1, H_out, W_out]
    wd = tf.broadcast_to(wd, [N, C, H_out, W_out])

    # print("wa = ",  wa)
    # print("wb = ",  wb)
    # print("wc = ",  wc)
    # print("wd = ",  wd)

    output = Ia * wa + Ib * wb + Ic * wc + Id * wd

    return output




















# def torch_nn_functional_grid_sample2(image, grid, mode='bilinear', padding_mode="zeros", align_corners=False):


#     def grid_sampler_unnormalize_tf(coord, side, align_corners):
#         if align_corners:
#             return ((coord + 1) / 2) * (side - 1)
#         else:
#             return ((coord + 1) * side - 1) / 2
            
#     def grid_sampler_compute_source_index_tf(coord, size, align_corners):
#         return grid_sampler_unnormalize_tf(coord, size, align_corners)

#     def safe_get_tf(image, n, c, x, y, H, W):
#         value = tf.zeros([1])
#         x = tf.cast(x, tf.int32)  # Ensure x is an integer type
#         y = tf.cast(y, tf.int32)  # Ensure y is an integer type
#         if x >= 0 and x < W and y >= 0 and y < H:
#             value = image[n, c, y, x]
#         return value


#     assert padding_mode == "zeros", "only zeros padding_mode is implemented right now"
#     assert padding_mode == "zeros", "only zeros padding_mode is implemented right now"
#     assert mode == "bilinear", "only bilinear is implemented right now"
#     assert len(image.shape) == 4, "len(input.shape) must be 4"

#     N, C, H_in, W_in = image.shape

#     H_out = grid.shape[1]
#     W_out = grid.shape[2]
    
#     # output_tensor = tf.zeros_like(image)
    
#     output_tensor = np.zeros( [N, C, H_out, W_out] )
#     # np.zeros_like(image)


#     np_Ia = np.zeros( [N, C, H_out, W_out] )
#     np_Ib = np.zeros( [N, C, H_out, W_out] )
#     np_Ic = np.zeros( [N, C, H_out, W_out] )
#     np_Id = np.zeros( [N, C, H_out, W_out] )

#     np_x0 = np.zeros( [N, W_out, H_out] )
#     np_y0 = np.zeros( [N, W_out, H_out] )

#     np_x1 = np.zeros( [N, W_out, H_out] )
#     np_y1 = np.zeros( [N, W_out, H_out] )



#     np_wa = np.zeros( [N, C, H_out, W_out] )
#     np_wb = np.zeros( [N, C, H_out, W_out] )
#     np_wc = np.zeros( [N, C, H_out, W_out] )
#     np_wd = np.zeros( [N, C, H_out, W_out] )

#     for n in range(N):
#         for w in range(W_out):
#             for h in range(H_out):
#                 # Get corresponding grid x and y
#                 x = grid[n, h, w, 1]
#                 y = grid[n, h, w, 0]
                
#                 # Unnormalize with align_corners condition
#                 ix = grid_sampler_compute_source_index_tf(x, W_in, align_corners)
#                 iy = grid_sampler_compute_source_index_tf(y, H_in, align_corners)
                
#                 x0 = tf.floor(ix)
#                 np_x0[n, h, w] = x0.numpy()

#                 x1 = x0 + 1

#                 np_x1[n, h, w] = x1.numpy()


#                 y0 = tf.floor(iy)
#                 np_y0[n, h, w] = y0.numpy()


#                 y1 = y0 + 1
    
#                 np_y1[n, h, w] = y1.numpy()

#                 # Get W matrix before I matrix, as I matrix requires Channel information
#                 wa = (x1 - ix) * (y1 - iy)
#                 wb = (x1 - ix) * (iy - y0)
#                 wc = (ix - x0) * (y1 - iy)
#                 wd = (ix - x0) * (iy - y0)

                
#                 # Get values of the image by provided x0, y0, x1, y1 by channel
#                 for c in range(C):
#                     Ia = safe_get_tf(image, n, c, y0, x0, H_in, W_in)
#                     Ib = safe_get_tf(image, n, c, y1, x0, H_in, W_in)
#                     Ic = safe_get_tf(image, n, c, y0, x1, H_in, W_in)
#                     Id = safe_get_tf(image, n, c, y1, x1, H_in, W_in)
#                     # out_ch_val = Ia * wa + Ib * wb + Ic * wc + Id * wd

#                     # output_tensor[n, h, w, c] = out_ch_val
#                     # output_tensor[n, h, w, c] = out_ch_val.numpy()

#                     np_Ia[n, c, h, w] = Ia.numpy()
#                     np_Ib[n, c, h, w] = Ib.numpy()
#                     np_Ic[n, c, h, w] = Ic.numpy()
#                     np_Id[n, c, h, w] = Id.numpy()

#                     np_wa[n, c, h, w] = wa.numpy()
#                     np_wb[n, c, h, w] = wb.numpy()
#                     np_wc[n, c, h, w] = wc.numpy()
#                     np_wd[n, c, h, w] = wd.numpy()

#                     # output_tensor[n, c, h, w] = out_ch_val.numpy()

#     # print("np_Ia = ", np_Ia)
#     # print("np_Ib = ", np_Ib)
#     # print("np_Ic = ", np_Ic)
#     # print("np_Id = ", np_Id)
#     print("np_wa = ", np_wa)
#     print("np_wb = ", np_wb)
#     print("np_wc = ", np_wc)
#     print("np_wd = ", np_wd)

#     # print("np_Ia.shape = ", np_Ia.shape)

#     print("np_x1 = ", np_x1)

#     print("np_y1 = ", np_y1)

#     # print("np_x0 = ", np_x0)
#     # print("np_x0.shape = ", np_x0.shape)

#     # print("np_y0 = ", np_y0)
#     # print("np_y0.shape = ", np_y0.shape)

#     # print("np_Ib = ", np_Ib)
#     # print("np_Ic = ", np_Ic)
#     # print("np_Id = ", np_Id)


#     output_tensor = tf.convert_to_tensor(output_tensor)
#     return output_tensor



































class Normal:
    def __init__(self, loc, scale):
        # #mean
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

        import tensorflow_probability as tfp

        self.distribution = tfp.distributions.Normal(loc=loc, scale=scale, name='Normal')

        self.first_rsample = True


    def log_prob(self, x):
        """
        Computes the log of the probability density function of a normal distribution.

        Args:
            x: The point at which to compute the probability density.
            mean: The mean of the normal distribution.
            std: The standard deviation of the normal distribution.

        Returns:
            Log of the probability density.
        """
        # # var = self.scale**2
        # log_pdf = -tf.math.log(self.scale * tf.math.sqrt(2 * tf.constant(np.pi))) - 0.5 * ((x - self.loc) ** 2) / (self.scale ** 2)

        # # log_pdf = torch.tensor(log_pdf.numpy())
        
        # return log_pdf

        return self.distribution.log_prob(x)


    def sample(self, sample_shape = tf.TensorShape([])):
        """
        Sample from a normal distribution.
        """

        # if OUTPUT_VARIABLES:
        #     print("1sample.shape = ", shape)

        # if shape == None or shape == tf.TensorShape([]):
        #     shape = self.loc.shape

        # if OUTPUT_VARIABLES:
        #     print("1sample.shape = ", shape)

        # sampled = tf.random.normal(shape=shape, mean=self.loc, stddev=self.scale)

        # print("Normal.sample(): = ", sampled)

        # # sampled = torch.tensor(sampled.numpy())

        # # print("normal: sampled = ", sampled)

        # return sampled
        return self.distribution.sample(sample_shape)


    def rsample(self, sample_shape = tf.TensorShape([])):

        if self.first_rsample:
            import tensorflow_probability as tfp
            self.distribution = tfp.distributions.Normal(loc=tf.constant(0.0), scale=tf.constant(1.0))
            self.first_rsample = False
        eps = self.distribution.sample(sample_shape)
        result = self.loc + eps * self.scale
        return result


    def entropy(self):
        """
        Computes the entropy of the normal distribution.

        Returns:
            The entropy of the normal distribution.
        """
        # Using the formula H(X) = 0.5 * log(2 * pi * e * std^2)
        # entropy = 0.5 * tf.math.log(2 * tf.constant(np.pi) * tf.constant(np.e) * self.scale ** 2)
        
        # # entropy = torch.tensor(entropy.numpy())

        # return entropy

        return self.distribution.entropy()


# import tensorflow as tf

def _sum_rightmost(x, n):
    """
    Sum the last n dimensions of a tensor.
    
    Args:
        x: The input tensor.
        n: The number of last n dimensions to sum over.
        Returns:
            The tensor after summing.
    """
    # Get the total number of dimensions of the tensor
    num_dims = len(x.shape)
    
    # The dimensions to sum over are the last n dimensions
    axes = list(range(num_dims - n, num_dims))
    
    # Use tf.reduce_sum to sum over the specified dimensions
    return tf.reduce_sum(x, axis=axes)



class Independent:
    def __init__(self, base_distribution, reinterpreted_batch_ndims, validate_args=None):
        if reinterpreted_batch_ndims > len(base_distribution.batch_shape):
            raise ValueError(
                "Expected reinterpreted_batch_ndims <= len(base_distribution.batch_shape), "
                f"actual {reinterpreted_batch_ndims} vs {len(base_distribution.batch_shape)}"
            )
        # shape = base_distribution.batch_shape + base_distribution.event_shape
        # # print("shape = ", shape)

        # # if base_distribution.event_shape != :
        # event_dim = reinterpreted_batch_ndims + len(base_distribution.event_shape)
        # # print("event_dim = ", event_dim)
        # # print("reinterpreted_batch_ndims = ", reinterpreted_batch_ndims)
        # # print("len(base_distribution.event_shape) = ", len(base_distribution.event_shape))

        # self.batch_shape = shape[: len(shape) - event_dim]
        # self.event_shape = shape[len(shape) - event_dim :]

        # # print("self.batch_shape = ", self.batch_shape)
        # # print("self.event_shape = ", self.event_shape)

        # self.base_dist = base_distribution
        # self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        # # super().__init__(batch_shape, event_shape, validate_args=validate_args)

        import tensorflow_probability as tfp

        # print("base_distribution = ", base_distribution)

        # print("base_distribution.distribution = ", base_distribution.distribution)

        self.distribution = tfp.distributions.Independent(base_distribution.distribution, reinterpreted_batch_ndims=reinterpreted_batch_ndims)  


    def log_prob(self, value):
        # log_prob = self.base_dist.log_prob(value)
        # # print("log_prob before = ", log_prob)
        # return _sum_rightmost(log_prob, self.reinterpreted_batch_ndims)

        return self.distribution.log_prob(value)

    def entropy(self):
        # entropy = self.base_dist.entropy()
        # return _sum_rightmost(entropy, self.reinterpreted_batch_ndims)
        return self.distribution.entropy()
    
    def sample(self, sample_shape=tf.TensorShape([])):
        # sample_result = self.base_dist.sample(sample_shape)
        # print("Independent: sample_result = ", sample_result)
        # return sample_result
        return self.distribution.sample(sample_shape)
    

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

        # print("Categorical: __init__(): probs = ", probs)

        # print("Categorical: __init__(): logits = ", logits)


        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )


        import tensorflow_probability as tfp



        if probs is not None:
            self.probs = probs
            self.distribution = tfp.distributions.Categorical(probs=probs)  
        elif logits is not None:
            self.logits = logits
            self.probs = tf.nn.softmax(logits, axis=-1)
            self.distribution = tfp.distributions.Categorical(logits=logits)  
    

        # else:
        #     raise ValueError("Must specify either probs or logits.")
    
        # print("1: Categorical: __init__(): self.probs = ", self.probs)

        # # self.batch_shape = logits.shape

        # # self.event_shape = tf.TensorShape([])

        # # if self.probs is not None:
        # if len(self.probs.shape) < 1:
        #     raise ValueError("`probs` parameter must be at least one-dimensional.")
        # # self.probs = probs / probs.sum(-1, keepdim=True)
        # if probs is not None:
        #     self.probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
        
        # print("2: Categorical: __init__(): self.probs = ", self.probs)

        # # else:
        # #     raise ValueError("must specify probs.")
        #     # if logits.dim() < 1:
        #     #     raise ValueError("`logits` parameter must be at least one-dimensional.")
        #     # Normalize
        #     # self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        # self._param = self.probs if probs is not None else self.logits
        # self._num_events = self._param.shape[-1]
        # # print("type(self._num_events) = ", type(self._num_events))
        # batch_shape = (
        #     self._param.shape[:-1] if len(self._param.shape) > 1 else tf.TensorShape([])
        # )
        # self.batch_shape = batch_shape
        # # super().__init__(batch_shape, validate_args=validate_args)

        # self.event_shape = tf.TensorShape([])



    def sample(self, sample_shape = tf.TensorShape([]) ):
        # # if shape == None or shape == tf.TensorShape([]):
        # #     shape = self.probs.shape

        # # return tf.random.categorical(self.probs, num_samples = 1, dtype=tf.int32)

        # if not isinstance(sample_shape, tf.TensorShape):
        #     sample_shape = tf.TensorShape(sample_shape)
        # num_elements = torch_tensor_item( tf.reduce_prod(sample_shape) )

        # print("Categorical: sample(): num_elements = ", num_elements)
        # # print("sample_shape.as_list() = ", sample_shape.as_list())
        # probs_2d = torch_reshape(self.probs, -1, self._num_events)

        # print("Categorical: sample(): probs_2d = ", probs_2d)

        # samples_2d = torch_tensor_transpose( torch_multinomial(probs_2d, num_elements, True), 0, 1)
        # extended_shape = tf.TensorShape(sample_shape + self.batch_shape + self.event_shape).as_list()

        # print("Categorical: sample(): extended_shape = ", extended_shape)

        # return torch_reshape( samples_2d, extended_shape )
        return self.distribution.sample(sample_shape)


    def log_prob(self, value):
        # assert len(value.shape.as_list()) <= 2
        # if self.probs is not None:

        #     value_shape_list = list(value.shape)

        #     batch_dim = value_shape_list[0]

        #     all_tensors = []

        #     for i in range(batch_dim):
        #         index = int(value[i, ...].numpy())  # Get the index

        #         log_prob_value = tf.gather(self.probs, index, axis=-1)  # Gather data from probs

        #         # Then compute the log
        #         log_prob_value = tf.math.log(log_prob_value)

        #         # Reshape the result
        #         all_tensors.append(tf.reshape(log_prob_value, [1, -1]))
                
        #     if batch_dim == 1:
        #         result = all_tensors[0]
        #     else:
        #         result = tf.concat(all_tensors, axis=0)

        #     return result

        # else:  # logits provided
        #     raise ValueError("Must specify probs.")

        return self.distribution.log_prob(value)


    def entropy(self):
        # return -tf.reduce_sum( self.probs * tf.math.log(self.probs), axis=-1 )
        return self.distribution.entropy()





class MixtureSameFamily:
    def __init__(
            self, mixture_distribution, component_distribution, validate_args=None
        ):
        # self._mixture_distribution = mixture_distribution
        # self._component_distribution = component_distribution

        import tensorflow_probability as tfp

        self.distribution = tfp.distributions.MixtureSameFamily(mixture_distribution = mixture_distribution.distribution, \
                                                                components_distribution = component_distribution.distribution)

        # if not isinstance(self._mixture_distribution, Categorical):
        #     raise ValueError(
        #         " The Mixture distribution needs to be an "
        #         " instance of torch.distributions.Categorical"
        #     )

        # # if not isinstance(self._component_distribution, Distribution):
        # #     raise ValueError(
        # #         "The Component distribution need to be an "
        # #         "instance of torch.distributions.Distribution"
        # #     )

        # # Check that batch size matches
        # mdbs = self._mixture_distribution.batch_shape

        # if OUTPUT_VARIABLES:
        #     print("self._component_distribution.batch_shape = ", self._component_distribution.batch_shape)

        # cdbs = self._component_distribution.batch_shape[:-1]
        # # cdbs = self._component_distribution.batch_shape
        
        # for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
        #     if size1 != 1 and size2 != 1 and size1 != size2:
        #         raise ValueError(
        #             f"`mixture_distribution.batch_shape` ({mdbs}) is not "
        #             "compatible with `component_distribution."
        #             f"batch_shape`({cdbs})"
        #         )

        # # Check that the number of mixture component matches
        # km = self._mixture_distribution.logits.shape[-1]
        # kc = self._component_distribution.batch_shape[-1]
        # if km is not None and kc is not None and km != kc:
        #     raise ValueError(
        #         f"`mixture_distribution component` ({km}) does not"
        #         " equal `component_distribution.batch_shape[-1]`"
        #         f" ({kc})"
        #     )
        # self._num_component = km

        # event_shape = self._component_distribution.event_shape
        # self._event_ndims = len(event_shape)

        # self.batch_shape = cdbs
        # self.event_shape = event_shape

        # # super().__init__(
        # #     batch_shape=cdbs, event_shape=event_shape, validate_args=validate_args
        # # )


    def log_prob(self, x):
        # # if self._validate_args:
        # #     self._validate_sample(x)
        # x = tf.expand_dims(x, axis=-1 - self._event_ndims)
        # log_prob_x = self._component_distribution.log_prob(x)  # [S, B, k]


        # log_mix_prob = tf.math.log(self._mixture_distribution.probs)

        # return torch_logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]
        return self.distribution.log_prob(x)


    def sample(self, sample_shape = tf.TensorShape([]) ):
        # with torch_no_grad() as tape:
        #     # sample_len = len(sample_shape)
        #     # batch_len = len(self.batch_shape)

        #     # sample_len = sample_shape[0]
        #     # batch_len = self.batch_shape[0]
        #     sample_len = len(sample_shape)
        #     # [0] if sample_shape.rank > 0 else 0
        #     batch_len = len(self.batch_shape)
        #     # [0] if self.batch_shape.rank > 0 else 0



        #     gather_dim = sample_len + batch_len
        #     es = self.event_shape



        #     # mixture samples [n, B]
        #     mix_sample = self._mixture_distribution.sample(sample_shape)
        #     mix_shape = mix_sample.shape

        #     # component samples [n, B, k, E]
        #     comp_samples = self._component_distribution.sample(sample_shape)

        #     mix_sample_shape = list(mix_shape) + [1] * (len(es) + 1)

        #     print("MixtureSameFamily.sample(): list(mix_shape) = ", list(mix_shape))

        #     print("MixtureSameFamily.sample(): [1] * (len(es) + 1) = ", [1] * (len(es) + 1))

        #     print("MixtureSameFamily.sample(): mix_sample_shape = ", mix_sample_shape)

        #     # Gather along the k dimension
        #     mix_sample_r = torch_reshape(mix_sample,
        #         mix_sample_shape
        #     )

        #     shape_list = [1] * len(mix_shape) + [1] + es

        #     print("MixtureSameFamily.sample(): mix_sample_r = ", mix_sample_r)
        #     print("MixtureSameFamily.sample(): shape_list = ", shape_list)

        #     mix_sample_r = torch_tensor_repeat( mix_sample_r,
        #         shape_list
        #     )

        #     # print("mix_sample_r:2 = ", mix_sample_r)

        #     print("MixtureSameFamily.sample(): comp_samples = ", comp_samples)
        #     print("MixtureSameFamily.sample(): gather_dim = ", gather_dim)
        #     print("MixtureSameFamily.sample(): mix_sample_r = ", mix_sample_r)

        #     # (40, 5, 8)
        #     # 40
        #     # (40, 1, 8)

        #     samples = torch_gather(comp_samples, gather_dim, mix_sample_r)

        #     return torch_squeeze(samples, gather_dim)
        return self.distribution.sample(sample_shape)












# class MultivariateNormal:
#     def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):

#         import tensorflow_probability as tfp


#         self.distribution = tfp.distributions.Independent(loc, covariance_matrix)  


#     def log_prob(self, value):
#         # log_prob = self.base_dist.log_prob(value)
#         # # print("log_prob before = ", log_prob)
#         # return _sum_rightmost(log_prob, self.reinterpreted_batch_ndims)

#         return self.distribution.log_prob(value)

#     def entropy(self):
#         # entropy = self.base_dist.entropy()
#         # return _sum_rightmost(entropy, self.reinterpreted_batch_ndims)
#         return self.distribution.entropy()
    
#     def sample(self, sample_shape=tf.TensorShape([])):
#         # sample_result = self.base_dist.sample(sample_shape)
#         # print("Independent: sample_result = ", sample_result)
#         # return sample_result
#         return self.distribution.sample(sample_shape)
    