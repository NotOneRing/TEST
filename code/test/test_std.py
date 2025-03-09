import torch
import tensorflow as tf
import numpy as np

# # define custom torch_std function
# def torch_std(input, dim=None, *, correction=1, keepdim=False, out=None):
#     input = tf.convert_to_tensor(input, dtype=tf.float32)

#     # calculate mean
#     mean = tf.reduce_mean(input, axis=dim, keepdims=True)

#     # calculate variance
#     variance = tf.reduce_mean(tf.square(input - mean), axis=dim, keepdims=keepdim)

#     # apply Bessel's correction
#     if correction != 0:
#         count = tf.shape(input)[dim] if dim is not None else tf.size(input)
#         count = tf.cast(count, tf.float32)
#         variance *= count / (count - correction)

#     # return standard deviation
#     return tf.sqrt(variance)

from util.torch_to_tf import torch_std


def test_std():
    # test data
    data = np.random.rand(3, 4).astype(np.float32)
    print("Input data:\n", data)

    # calculate std in PyTorch
    pytorch_tensor = torch.tensor(data)
    pytorch_result = torch.std(pytorch_tensor, dim=1, correction=1, keepdim=True)
    # print("PyTorch std result:\n", pytorch_result.numpy())

    # calculate std in TensorFlow
    tensorflow_tensor = tf.convert_to_tensor(data)
    tensorflow_result = torch_std(tensorflow_tensor, dim=1, correction=1, keepdim=True)
    # print("TensorFlow std result:\n", tensorflow_result.numpy())

    # compare results
    if np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6):
        print("The results match!")
    else:
        print("The results do not match!")

    assert np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6)

    # calculate std in PyTorch
    pytorch_tensor = torch.tensor(data)
    pytorch_result = torch.std(pytorch_tensor, dim=1, correction=0, keepdim=True)
    # print("PyTorch std result:\n", pytorch_result.numpy())

    # calculate std in TensorFlow
    tensorflow_tensor = tf.convert_to_tensor(data)
    tensorflow_result = torch_std(tensorflow_tensor, dim=1, correction=0, keepdim=True)
    # print("TensorFlow std result:\n", tensorflow_result.numpy())

    # compare results
    if np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6):
        print("The results match!")
    else:
        print("The results do not match!")

    assert np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6)


    # calculate std in PyTorch
    pytorch_tensor = torch.tensor(data)
    pytorch_result = torch.std(pytorch_tensor, dim=1, correction=0.5, keepdim=True)
    # print("PyTorch std result:\n", pytorch_result.numpy())

    # calculate std in TensorFlow
    tensorflow_tensor = tf.convert_to_tensor(data)
    tensorflow_result = torch_std(tensorflow_tensor, dim=1, correction=0.5, keepdim=True)
    # print("TensorFlow std result:\n", tensorflow_result.numpy())

    # compare results
    if np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6):
        print("The results match!")
    else:
        print("The results do not match!")

    assert np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6)

    # calculate std in PyTorch
    pytorch_tensor = torch.tensor(data)
    pytorch_result = torch.std(pytorch_tensor, correction=0.5, keepdim=True)
    # print("PyTorch std result:\n", pytorch_result.numpy())

    # calculate std in TensorFlow
    tensorflow_tensor = tf.convert_to_tensor(data)
    tensorflow_result = torch_std(tensorflow_tensor, correction=0.5, keepdim=True)
    # print("TensorFlow std result:\n", tensorflow_result.numpy())

    # compare results
    if np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6):
        print("The results match!")
    else:
        print("The results do not match!")

    assert np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6)


    # calculate std in PyTorch
    pytorch_tensor = torch.tensor(data)
    pytorch_result = torch.std(pytorch_tensor, correction=0.5)
    # print("PyTorch std result:\n", pytorch_result.numpy())

    # calculate std in TensorFlow
    tensorflow_tensor = tf.convert_to_tensor(data)
    tensorflow_result = torch_std(tensorflow_tensor, correction=0.5)
    # print("TensorFlow std result:\n", tensorflow_result.numpy())

    # compare results
    if np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6):
        print("The results match!")
    else:
        print("The results do not match!")


    assert np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6)



    # # calculate std in PyTorch
    # pytorch_tensor = torch.tensor(data)
    # torch.std(pytorch_tensor, correction=1, out = pytorch_result)
    # print("PyTorch std result:\n", pytorch_result.numpy())

    # # calculate std in TensorFlow
    # tensorflow_tensor = tf.convert_to_tensor(data)
    # torch_std(tensorflow_tensor, correction=1, out = tensorflow_result)
    # print("TensorFlow std result:\n", tensorflow_result.numpy())

    # # compare results
    # if np.allclose(pytorch_result.numpy(), tensorflow_result.numpy(), atol=1e-6):
    #     print("The results match!")
    # else:
    #     print("The results do not match!")





test_std()





