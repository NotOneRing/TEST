import tensorflow as tf
import numpy as np
from collections import OrderedDict



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


    input_dim_list = list(input_tensor.shape)
    dim_list = list(index_tensor.shape)

    #transfer negative index to positive one
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




def torch_quantile(input_tensor, q, dim=None, keepdim=False, interpolation='linear'):
    """
    Compute the quantile of the input_tensor along a specified axis using TensorFlow.
    """
    
    dim_None_flag = False

    input_shape_list = input_tensor.shape.as_list()
    if isinstance(q, int):
        q_shape_list = []
    else:
        q_shape_list = q.shape.as_list()


    # Flatten the input if axis is None
    if dim is None:
        input_tensor = tf.reshape(input_tensor, [-1])
        dim = 0
        dim_None_flag = True

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

    
    if dim_None_flag:
        if keepdim:
            result_shape = input_shape_list
            for i in range(len(result_shape)):
                result_shape[i] = 1
            result_shape = q_shape_list + result_shape

            result = tf.reshape( result, result_shape )
        else:
            if isinstance(q, int) or (len(q_shape_list) == 1 and q_shape_list[0] == 1):
                result = tf.reshape( result, [])
            else:
                result = tf.reshape(result, q_shape_list)

    elif keepdim==False:
        result = torch_tensor_transpose(result, 0, dim)

        if isinstance(q, int) or (len(q_shape_list) == 1 and q_shape_list[0] == 1):
            result_shape = input_shape_list[:dim] + input_shape_list[dim+1:]
        else:
            result_shape = q_shape_list + input_shape_list[:dim] + input_shape_list[dim+1:]

        result = tf.reshape( result, result_shape )
    elif keepdim==True:
        result = torch_tensor_transpose(result, 0, dim)

        input_shape_list[dim] = 1
        if isinstance(q, int) or (len(q_shape_list) == 1 and q_shape_list[0] == 1):
            result_shape = input_shape_list
        else:
            result_shape = [q_shape_list[0], ] + input_shape_list

        result = tf.reshape( result, result_shape )

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
    '''
    torch.arange(start=0, end, step=1, *, 
    out=None, dtype=None, layout=torch.strided, 
    device=None, requires_grad=False)'
    '''
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
    logits = tf.math.log(input)
    logits = logits[None, :]
    return tf.random.categorical(logits, num_samples=num_samples, dtype=tf.int32).numpy().flatten()



def torch_where(index_tensor, input_tensor = None, replace_value = None):
    if input_tensor != None and replace_value != None:
        result = tf.where(index_tensor, input_tensor, replace_value)
    else:
        result = tf.where(index_tensor, input_tensor, replace_value)
        assert len(result.shape) == 2, "result's shape length must be two"
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
        input.assign( temp_variable )
    else:
        raise RuntimeError("Input must be tf.Variable to be able to changed")








def torch_zeros(*size, dtype=tf.float32):
    '''
    torch.zeros(size, dtype=torch.float32, device=None, requires_grad=False)
    '''
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




def torch_full(size, fill_value, dtype=None):
    '''
    torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    '''
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
        return tf.meshgrid(*tensors[0], indexing=indexing)

    else:
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
    '''
    Wrapper for torch.randn_like(x)
    Returns a tensor with the same size as input 
    that is filled with random numbers from a normal
    distribution with mean 0 and variance 1. 
    Please refer to torch.randn() for the 
    sampling process of complex dtypes. 
    torch.randn_like(input) is equivalent 
    to torch.randn(input.size(), dtype=input.dtype, 
    layout=input.layout, device=input.device).
    '''
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








def torch_repeat_interleave(tensor, repeats, dim=None):

    
    if isinstance(repeats, int):
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


    if not isinstance(tensor, tf.Tensor):
        raise TypeError("Input must be a TensorFlow tensor.")
    if not repeats:
        raise ValueError("At least one repeat value must be provided.")


    if isinstance(repeats[0], (tuple, list)):
        repeat_shape = [ *repeats[0] ]
        repeats_tensor = tf.constant(repeats[0], dtype=tf.int32)
    else:
        repeat_shape = [*repeats]
        repeats_tensor = tf.constant(repeats, dtype=tf.int32)

    repeats_tensor = torch_reshape( repeats_tensor, -1)


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





from functools import wraps
import inspect
from typing import Callable, Union, Tuple, List, Dict, Any, Optional

def torch_vmap(func: Callable, 
         in_dims: Union[int, Tuple[int, ...], List[int], Dict[str, int]] = 0, 
         out_dims: Union[int, Tuple[int, ...], List[int], Dict[str, int]] = 0, 
         randomness: str = 'error', 
         *, 
         chunk_size: Optional[int] = None) -> Callable:
    """
    TensorFlow implementation of PyTorch's vmap function.
    
    vmap is the vectorizing map; vmap(func) returns a new function that maps func over some dimension 
    of the inputs. Semantically, vmap pushes the map into TensorFlow operations called by func, 
    effectively vectorizing those operations.
    
    Args:
        func (Callable): The function to be vectorized.
        in_dims (Union[int, Tuple[int, ...], List[int], Dict[str, int]], optional): 
            Specifies which dimension of each input should be mapped over. Default is 0.
            If an int, the same dimension is used for all inputs.
            If a tuple/list, each element corresponds to the dimension of the corresponding input.
            If a dict, keys are argument names and values are dimensions.
        out_dims (Union[int, Tuple[int, ...], List[int], Dict[str, int]], optional): 
            Specifies which dimension of the output the mapped dimension should appear. Default is 0.
            Format is the same as in_dims.
        randomness (str, optional): How to handle randomness in the function. Default is 'error'.
            Options are:
            - 'error': Raise an error if randomness is detected.
            - 'different': Use different random values for each batch element.
            - 'same': Use the same random values for each batch element.
        chunk_size (Optional[int], optional): If specified, the computation is chunked into smaller
            batches of the specified size. This can be useful for large batches to avoid memory issues.
            Default is None (no chunking).
    Returns:
        Callable: A vectorized version of the input function.
    
    Raises:
        ValueError: If randomness is set to 'error' and randomness is detected in the function.
        ValueError: If in_dims or out_dims have invalid values.
        NotImplementedError: If certain features are not yet implemented.
    """
    
    if randomness not in ['error', 'different', 'same']:
        raise ValueError(f"randomness must be one of 'error', 'different', 'same', got {randomness}")
    
    
    # Handle different types of in_dims and out_dims
    def normalize_dims(dims, arg_names):
        if isinstance(dims, int):
            return {name: dims for name in arg_names}
        elif isinstance(dims, (list, tuple)):
            if len(dims) != len(arg_names):
                raise ValueError(f"Length of dims {len(dims)} does not match number of arguments {len(arg_names)}")
            return {name: dim for name, dim in zip(arg_names, dims)}
        elif isinstance(dims, dict):
            # Ensure all keys in dims are in arg_names
            for key in dims:
                if key not in arg_names:
                    raise ValueError(f"Argument '{key}' not found in function signature")
            # Fill in default value (0) for missing keys
            return {name: dims.get(name, 0) for name in arg_names}
        else:
            raise ValueError(f"dims must be an int, tuple, list, or dict, got {type(dims)}")
    


    @wraps(func)
    def wrapper(*args, **kwargs):



        # Get the argument names from the function signature
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        
        
        # Create a mapping of argument names to their values
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arg_dict = bound_args.arguments


        # Normalize in_dims and out_dims
        in_dims_dict = normalize_dims(in_dims, arg_dict.keys())

        # Handle chunking if specified
        if chunk_size is not None:

            # Determine the batch size from the first argument with a non-None in_dim
            batch_size = None
            for name, value in arg_dict.items():
                if in_dims_dict[name] is not None and isinstance(value, tf.Tensor):
                    dim = in_dims_dict[name]
                    if dim < 0:
                        dim = len(value.shape) + dim
                    batch_size = value.shape[dim]
                    break
            
            if batch_size is None:
                raise ValueError("Could not determine batch size for chunking")
            
            # Split the computation into chunks
            results = []
            for i in range(0, batch_size, chunk_size):
                chunk_args = {}
                for name, value in arg_dict.items():
                    if in_dims_dict[name] is not None and isinstance(value, tf.Tensor):
                        dim = in_dims_dict[name]
                        if dim < 0:
                            dim = len(value.shape) + dim
                        indices = [slice(None)] * len(value.shape)
                        indices[dim] = slice(i, min(i + chunk_size, batch_size))
                        chunk_args[name] = value[tuple(indices)]
                    else:
                        chunk_args[name] = value
                
                
                # Call the function with the chunked arguments
                chunk_result = func(**chunk_args)

                results.append(chunk_result)
            
            # Concatenate the results along the out_dims
            if isinstance(out_dims, int):
                return tf.concat(results, axis=out_dims)
            else:
                # Handle more complex out_dims structures
                # This is a simplified implementation and may need to be extended
                if isinstance(chunk_result, tuple):
                    return tuple( tf.concat([r[i] for r in results], axis=out_dims[i]) 
                                for i in range(len(chunk_result)) 
                                )
                elif isinstance(chunk_result, dict):
                    return {k: tf.concat([r[k] for r in results], axis=out_dims.get(k, 0)) 
                           for k in chunk_result}
                else:
                    return tf.concat(results, axis=out_dims)
        
        # Handle randomness
        if randomness == 'error':
            # In a real implementation, we would check for random operations here
            # For simplicity, we'll just warn that this check is not implemented
            tf.print("Warning: randomness='error' check is not fully implemented")
        
        # Vectorize the function using tf.vectorized_map or manual broadcasting
        # This is a simplified implementation that handles basic cases
        
        # Prepare the inputs for vectorization
        vectorized_args = {}
        batch_size = None
        batch_dim_indices = {}
        
        for name, value in arg_dict.items():
            dim = in_dims_dict[name]
            if dim is None or not isinstance(value, tf.Tensor):
                # Non-tensor arguments or None in_dims are passed as-is
                vectorized_args[name] = value
                continue
            
            # Normalize negative dimensions
            if dim < 0:
                dim = len(value.shape) + dim
            
            # Record the batch dimension for later use
            if batch_size is None:
                batch_size = value.shape[dim]
            elif value.shape[dim] != batch_size:
                raise ValueError(f"Inconsistent batch sizes: got {value.shape[dim]} for argument '{name}', "
                                f"expected {batch_size}")
            
            # Move the batch dimension to the first position for vectorized_map
            if dim != 0:
                perm = list(range(len(value.shape)))
                perm.pop(dim)
                perm.insert(0, dim)
                value = tf.transpose(value, perm)
            
            vectorized_args[name] = value
            batch_dim_indices[name] = dim
        
        if batch_size is None:
            # No batch dimensions found, just call the function directly

            return func(**arg_dict)
        
        # Define a function that operates on a single slice of the batch
        def apply_func(batch_indices):
            single_args = {}
            for name, value in arg_dict.items():
                if name in batch_dim_indices:
                    dim = batch_dim_indices[name]
                    if dim == 0:
                        # If the batch dimension is already at index 0, just index it
                        indices = [batch_indices] + [slice(None)] * (len(value.shape) - 1)
                        single_args[name] = value[tuple(indices)]
                    else:
                        # If we transposed the tensor earlier, index the first dimension
                        indices = [batch_indices] + [slice(None)] * (len(value.shape) - 1)
                        single_args[name] = vectorized_args[name][tuple(indices)]
                else:
                    # Non-batched arguments are passed as-is
                    single_args[name] = value
            

            return func(**single_args)
        
        # Use tf.vectorized_map to apply the function to each batch element
        batch_indices = tf.range(batch_size)

        result = tf.vectorized_map(lambda i: apply_func(i), batch_indices)
        
        # Handle out_dims
        # For simplicity, we'll assume out_dims is an int and the result is a tensor
        # A more complete implementation would handle complex output structures
        if isinstance(out_dims, int) and out_dims != 0 and isinstance(result, tf.Tensor):
            # Move the batch dimension (currently at 0) to the specified out_dims
            ndim = len(result.shape)
            if out_dims < 0:
                cur_out_dims = ndim + out_dims
            else:
                cur_out_dims = out_dims

            if out_dims >= ndim:
                raise ValueError(f"out_dims {out_dims} is out of bounds for tensor of rank {ndim}")
            
            perm = list(range(ndim))
            perm.pop(0)
            perm.insert(cur_out_dims, 0)
            result = tf.transpose(result, perm)
        
        return result
    
    return wrapper










def torch_func_stack_module_state(models):
    # Stack all model parameters and buffers

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
                get_result = trainable.get(matched_name)
                if get_result != None:
                    cur_tensor = tf.convert_to_tensor(var)
                    cur_tensor = torch_unsqueeze(cur_tensor, 0)
                    trainable[matched_name] = torch_cat([tf.convert_to_tensor(trainable[matched_name]), cur_tensor], 0)
                else:
                    cur_tensor = tf.convert_to_tensor(var)
                    cur_tensor = torch_unsqueeze(cur_tensor, 0)
                    trainable[matched_name] = cur_tensor
            else:
                if OUTPUT_VARIABLES:
                    print("var.name = ", var.name)
                raise RuntimeError("Network name not recognized!")
        for var in model.non_trainable_variables:
            # Extract last part after the last '/'
            match = re.search(pattern, var.name)
            if match:
                matched_name = match.group()
                get_result = non_trainable.get(matched_name)
                if get_result != None:
                    cur_tensor = tf.convert_to_tensor(var)
                    cur_tensor = torch_unsqueeze(cur_tensor, 0)
                    non_trainable[matched_name] = torch_cat([tf.convert_to_tensor(non_trainable[matched_name]), cur_tensor], 0)
                else:
                    cur_tensor = tf.convert_to_tensor(var)
                    cur_tensor = torch_unsqueeze(cur_tensor, 0)
                    non_trainable[matched_name] = cur_tensor
            else:
                if OUTPUT_VARIABLES:
                    print("var.name = ", var.name)
                raise RuntimeError("Network name not recognized!")

    return trainable, non_trainable









def torch_func_functional_call(model, params, x):
    # Performs a functional call on the module by replacing the module parameters and buffers with the provided ones.
    from copy import deepcopy
    former_params = deepcopy(model.trainable_variables)

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
    
    # Assign ones to the variable
    tensor.assign(tf.ones_like(tensor))







def torch_nn_init_xavier_normal_(tensor, gain):

    assert len(tensor.shape) == 2, "input tensor must be of shape 2"
    fan_in = tensor.shape[0]
    fan_out = tensor.shape[1]

    std = gain * tf.sqrt( 2 / (fan_in + fan_out) )

    # Use TensorFlow random number generator
    normal_values = tf.random.normal(
        shape=tensor.shape,
        mean=0.0,
        stddev=std,
        dtype=tensor.dtype  # Automatically match the variable's data type (e.g., float32)
    )

    # Directly assign the generated TensorFlow tensor
    tensor.assign(normal_values)





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
        result = tf.clip_by_value(x, float('-inf'), 20)

        beta = 1
        result = beta * result

        result = tf.math.softplus(result)

        return x * tf.math.tanh(result)

    
    def get_config(self):
        config = super(nn_Mish, self).get_config()  # Call the parent layer's get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_Mish: from_config()")
        result = cls(**config)
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
                 device=None, dtype=None, 
                 name_Dense = None, model=None, **kwargs):
        
        super(nn_Linear, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features

        self.device = device




        if model == None:

            import math

            self.model = tf.keras.layers.Dense(
                out_features,
                activation=None,
                use_bias=True,
                kernel_initializer = tf.keras.initializers.Constant(
                    pytorch_weight_initializer((in_features, out_features), in_features)),
                bias_initializer = tf.keras.initializers.Constant(
                    pytorch_bias_initializer((out_features,), in_features)),
                dtype=dtype,
                name = name_Dense
            )

            
        else:
            self.model = model


        if OUTPUT_VARIABLES:
            print("nn_Linear: name_Dense = ", name_Dense)
            print("nn_Linear: self.model.name = ", self.model.name)

    def get_config(self):
        # Get the configuration of the layer and return it as a dictionary
        config = super(nn_Linear, self).get_config()  # Call the parent layer's get_config()
        config.update({
            "in_features": self.in_features,
            "out_features": self.out_features,
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
        return result


    def call(self, x):

        result = self.model(x)


        if OUTPUT_VARIABLES and DEBUG and self.model.built:
            weights = self.model.kernel
            bias = self.model.bias


            result1 = tf.matmul(x, weights) + bias  # broadcast addition


            print("nn_Linear.call() result1 = ", result1)


        return result


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






def nn_Parameter(data=None, requires_grad=True):
    if data is None:
        raise ValueError("data cannot be None. Please provide a tensor value.")
    return tf.Variable(data, trainable=requires_grad, name="nn_parameter")



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
                 model_list = None, 
                 model = None, 
                 **kwargs):
        super(nn_Sequential, self).__init__(
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

    
    
    def call(self, x):
        output = x

        output = self.model(output)
        
        return output
    
    def __getitem__(self, id):
        return self.model.layers[id]

    def __iter__(self):
        return iter(self.model.layers)

    def __len__(self):
            nn_Sequential_len = len(self.model.layers)
            return nn_Sequential_len
    

    def get_config(self):
        # Get the configuration of all layers in the model_list
        config = super(nn_Sequential, self).get_config()  # Call the parent class get_config()
        
        
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

        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
            'DiffusionMLP': DiffusionMLP,
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
        # Register custom class with Keras
        get_custom_objects().update(cur_dict)


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

        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
            'DiffusionMLP': DiffusionMLP,
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
        # Register custom class with Keras
        get_custom_objects().update(cur_dict)

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
            '_weight': self.embeddings.numpy() if hasattr(self.embeddings, 'numpy') else None
        })
        return config


    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("nn_Embedding: from_config()")
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
        config = super(nn_LayerNorm, self).get_config()
        config.update({
            'normalized_shape': self.normalized_shape,
            'eps': self.epsilon
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



    def call(self, x):
        """
        Forward pass for LayerNorm.

        Args:
            x (tf.Tensor): Input tensor to normalize.

        Returns:
            tf.Tensor: The normalized tensor.
        """

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
    def __init__(self, d_model, num_heads, name="nn_MultiheadAttention", batch_first = False, **kwargs):
        
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

        self.batch_first = batch_first

    def split_heads(self, x, batch_size):
        """Split the last dimension into multiple heads"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])


    def scaled_dot_product_attention(self, query, key, value, mask=None, attn_mask=None, key_padding_mask=None):
        """Compute scaled dot-product attention with support for multiple mask types"""
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply legacy mask if provided (for backward compatibility)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            # If attn_mask is 2D, expand it to match the batch size and number of heads
            if len(tf.shape(attn_mask)) == 2:
                attn_mask = tf.expand_dims(tf.expand_dims(attn_mask, 0), 0)
            
            # Apply the mask by adding a large negative value to masked positions
            scaled_attention_logits += (attn_mask * -1e9)
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask is expected to be of shape [batch_size, seq_len_k]
            # We need to reshape it to [batch_size, 1, 1, seq_len_k]
            key_padding_mask = tf.cast(key_padding_mask, tf.float32)
            key_padding_mask = tf.expand_dims(tf.expand_dims(key_padding_mask, 1), 2)
            
            # Apply the mask by adding a large negative value to masked positions
            scaled_attention_logits += (key_padding_mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights


    def call(self, query, key, value, mask=None, attn_mask=None, key_padding_mask=None):
        """
        Forward pass for MultiheadAttention with support for multiple mask types
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Legacy mask parameter (kept for backward compatibility)
            attn_mask: Optional mask applied to attention logits to mask specific position pairs
            key_padding_mask: Optional mask for key sequence to mask padded positions
        """
        if not self.batch_first:
            query_input = torch_tensor_transpose(query, 0, 1)
            key_input = torch_tensor_transpose(key, 0, 1)
            value_input = torch_tensor_transpose(value, 0, 1)
        else:
            query_input = query
            key_input = key
            value_input = value

        batch_size = tf.shape(query_input)[0]

        # Apply linear transformation to Q, K, V and split into multiple heads
        query_input = self.query_dense(query_input)
        key_input = self.key_dense(key_input)
        value_input = self.value_dense(value_input)

        query_input = self.split_heads(query_input, batch_size)
        key_input = self.split_heads(key_input, batch_size)
        value_input = self.split_heads(value_input, batch_size)

        # Scaled dot-product attention
        output, attention_weights = self.scaled_dot_product_attention(
            query_input, key_input, value_input, mask, attn_mask, key_padding_mask
        )

        # Concatenate the multiple heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        attention_weights = tf.reduce_mean(attention_weights, axis=1)  # take the average of all heads

        # Output transformation
        output = self.output_dense(output)

        if not self.batch_first:
            output = torch_tensor_transpose(output, 0, 1)
        else:
            output = output
        
        
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


    def call(self, x):
        """
        Forward pass for Conv1d.

        Args:
            x (tf.Tensor): Input tensor to convolve.

        Returns:
            tf.Tensor: The convolved tensor.
        """
        

        torch_to_tf_input = tf.transpose(x, perm = (0, 2, 1) )
        

        result = self.conv1d(torch_to_tf_input)

        result = tf.transpose(result, perm = (0, 2, 1) )

        return result



    def assign_torch_weights(self, numpy_torch_weights):
        torch_weight = numpy_torch_weights.transpose(2, 1, 0)  
        self.conv1d.kernel.assign(torch_weight)











class nn_Conv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups = 1, conv2d = None, **kwargs):
        super(nn_Conv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.stride = (stride, stride) if isinstance(stride, int) else stride

        self.bias = bias

        if conv2d:
            self.conv2d = conv2d
        else:
            # TensorFlow's Conv2D requires NHWC format
            self.conv2d = tf.keras.layers.Conv2D(
                filters=self.out_channels,
                kernel_size=self.kernel_size,
                strides=self.stride,
                padding="valid",  # PyTorch padding=0 corresponds to TensorFlow "valid"
                use_bias=self.bias,
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
        # Register custom class with Keras
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

        kernel_initializer = tf.keras.initializers.Constant(
            pytorch_weight_initializer((kernel_size, out_channels // groups, in_channels), in_features))

        bias_initializer = tf.keras.initializers.Constant(
            pytorch_bias_initializer((out_channels,), in_features) )


        self.conv1d_transpose = tf.keras.layers.Conv1DTranspose(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="same",
            use_bias=bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )

    def build(self, input_shape):
        self.conv1d_transpose.build((None, None, self.in_channels))


    def call(self, x):
    
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

        # Reshape the input tensor to group the channels
        batch_size, height, width, channels = tf.unstack(tf.shape(x))

        group_size = channels // self.num_groups

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
                 activation=nn_ReLU(), layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False):
        super(nn_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn_MultiheadAttention(d_model, nhead )
        self.multihead_attn = nn_MultiheadAttention(d_model, nhead )
        
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


    def call(self, tgt, memory, 
                tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                training=True):
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


    def apply_gradients(self, zipped_gradients):
        self.optimizer.apply_gradients(zipped_gradients)










def model_forward_backward_gradients(input_features, target_label, loss_func, model):

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

    global_norm = tf.norm(tf.stack([tf.norm(grad) for grad in grads_finite if grad is not None]))

    # Clip if the norm exceeds the maximum value
    clip_coef = max_norm / (global_norm + 1e-6)

    clip_coef_bf = tf.where(global_norm < max_norm, tf.ones_like(clip_coef), clip_coef)

    clipped_grads = [grad * clip_coef_bf if grad is not None else None for grad in grads]

    # If error_if_nonfinite is True, check for non-finite values
    if error_if_nonfinite:
        for g in grads:
            if g is not None and (tf.reduce_any(tf.is_nan(g)) or tf.reduce_any(tf.is_inf(g))):
                raise ValueError("Gradients contain non-finite values.")

    zip_gradients_params = zip(clipped_grads, parameters)

    optimizer.apply_gradients(zip_gradients_params)
    
    return clipped_grads











def torch_tensor_requires_grad_(tensor, requires_grad=True):
    if requires_grad:
        return tf.Variable(tensor, trainable=True)
    else:
        return tf.Variable(tensor, trainable=False)
    



































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
        self.data_size = len(dataset)


    def __iter__(self):
        """
        Iterator method, allows traversal of data using `for batch in dataloader`
        """

        rng = np.random.default_rng()  # Maintain reproducibility
        indices = list(range(self.data_size))

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

                    if value[0] is not None:
                        return_dict[key] = tf.convert_to_tensor(value)

                yield return_dict
                batch_data = {key: [] for key in batch_data}  # Reinitialize
                batch_count = 0

        if not self.drop_last and batch_count > 0:
            # Handle the last batch (if drop_last=False)

                return_dict = {}
                for key, value in batch_data.items():

                    if value[0] is not None:   
                        return_dict[key] = tf.convert_to_tensor(value)

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
        
    def step(self, epoch = None):
        import math

        #Because tensorflow automatically set epoch for each epoch，to achieve the same optimizer as the pytorch version, we choose to fix epoch=None manually
        if epoch is not None:
            epoch = int(epoch)  # Force conversion to integer


        epoch = None

        if epoch is None:

            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:

                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            
            if epoch >= self.first_cycle_steps:

                if self.cycle_mult == 1.0:

                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:

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

                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )

                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )

            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)


        if self.step_in_cycle == -1:
            self.lr = self.base_lr
        elif self.step_in_cycle < self.warmup_steps:
            self.lr = (self.max_lr - self.base_lr) * self.step_in_cycle / self.warmup_steps + self.base_lr
        else:
            self.lr = self.base_lr + (self.max_lr - self.base_lr) * ( 1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                ) / 2


        return self.lr









def safe_gather_nd(tensor, indices, default_value=0.0):

    tensor_len = len(tensor.shape)
    indices_last_len = tf.shape(indices)[-1]
    augment_dim = tensor_len - indices_last_len
    

    # Get tensor shape
    tensor_shape = tf.shape(tensor)
    max_indices = tensor_shape[:tf.shape(indices)[-1]]  # Calculate maximum index for each dimension

    # Check if indices are out of bounds
    is_out_of_bounds = tf.reduce_any(indices < 0, axis=-1) | tf.reduce_any(indices >= max_indices, axis=-1)

    match_dim_is_out_of_bounds = tf.expand_dims(is_out_of_bounds, axis=-1)  # (3, 1)
    

    # Create valid indices by replacing out-of-bounds indices with (0, 0, ...) to avoid errors
    safe_indices = tf.where(match_dim_is_out_of_bounds, tf.zeros_like(indices), indices)


    # Get gather_nd result
    gathered_values = tf.gather_nd(tensor, safe_indices)


    for i in range(augment_dim):
        is_out_of_bounds = tf.expand_dims(is_out_of_bounds, axis=-1)


    # Replace out-of-bounds indices with default_value
    result = tf.where(is_out_of_bounds, tf.fill(tf.shape(gathered_values),  tf.cast(default_value, gathered_values.dtype) ), gathered_values)

    return result


def torch_nn_functional_grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False):

    assert mode == "bilinear", "Only bilinear mode is supported."
    assert padding_mode == "zeros", "Only zeros padding_mode is implemented."

    # PyTorch: [N, C, H, W] → TensorFlow: [N, H, W, C]
    # x = tf.transpose(x, [0, 2, 3, 1])
    # N, H_in, W_in, C = x.shape
    N, C, H_in, W_in = image.shape

    H_out, W_out = grid.shape[1:3]

    # Normalize grid [-1, 1] → [0, H-1] or [0, W-1]
    def grid_sampler_unnormalize(coord, size, align_corners):
        if align_corners:
            return ((coord + 1) / 2) * (size - 1)
        else:
            return ((coord + 1) * size - 1) / 2

    ix = grid_sampler_unnormalize(grid[..., 1], W_in, align_corners)
    iy = grid_sampler_unnormalize(grid[..., 0], H_in, align_corners)
    

    x0 = tf.math.floor(ix)
    x1 = x0 + 1

    y0 = tf.math.floor(iy)
    y1 = y0 + 1

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

    
    Ia = tf.transpose(Ia, [0, 3, 1, 2])
    Ib = tf.transpose(Ib, [0, 3, 1, 2])
    Ic = tf.transpose(Ic, [0, 3, 1, 2])
    Id = tf.transpose(Id, [0, 3, 1, 2])

    wa = tf.expand_dims(wa, axis=1)  # Change to [N, 1, H_out, W_out]
    wa = tf.broadcast_to(wa, [N, C, H_out, W_out])
    wb = tf.expand_dims(wb, axis=1)  # Change to [N, 1, H_out, W_out]
    wb = tf.broadcast_to(wb, [N, C, H_out, W_out])
    wc = tf.expand_dims(wc, axis=1)  # Change to [N, 1, H_out, W_out]
    wc = tf.broadcast_to(wc, [N, C, H_out, W_out])
    wd = tf.expand_dims(wd, axis=1)  # Change to [N, 1, H_out, W_out]
    wd = tf.broadcast_to(wd, [N, C, H_out, W_out])


    output = Ia * wa + Ib * wb + Ic * wc + Id * wd

    return output


















































class Normal:
    def __init__(self, loc, scale):
        # #mean
        self.loc = loc

        #std
        self.scale = scale

        self.batch_shape = self.loc.shape

        self.event_shape = tf.TensorShape([])

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

        return self.distribution.log_prob(x)


    def sample(self, sample_shape = tf.TensorShape([])):
        """
        Sample from a normal distribution.
        """


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


        return self.distribution.entropy()


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

        import tensorflow_probability as tfp


        self.distribution = tfp.distributions.Independent(base_distribution.distribution, reinterpreted_batch_ndims=reinterpreted_batch_ndims)  


    def log_prob(self, value):

        return self.distribution.log_prob(value)

    def entropy(self):

        return self.distribution.entropy()
    
    def sample(self, sample_shape=tf.TensorShape([])):

        return self.distribution.sample(sample_shape)
    


class Categorical:
    def __init__(self, probs=None, logits=None):
        

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
    

    def sample(self, sample_shape = tf.TensorShape([]) ):

        return self.distribution.sample(sample_shape)


    def log_prob(self, value):

        return self.distribution.log_prob(value)


    def entropy(self):

        return self.distribution.entropy()





class MixtureSameFamily:
    def __init__(
            self, mixture_distribution, component_distribution, validate_args=None
        ):

        import tensorflow_probability as tfp

        self.distribution = tfp.distributions.MixtureSameFamily(mixture_distribution = mixture_distribution.distribution, \
                                                                components_distribution = component_distribution.distribution)


    def log_prob(self, x):

        return self.distribution.log_prob(x)


    def sample(self, sample_shape = tf.TensorShape([]) ):

        return self.distribution.sample(sample_shape)










