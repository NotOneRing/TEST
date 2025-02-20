import tensorflow as tf

import torch


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


def test_flatten():
    import numpy as np

    # Step 1: Create a NumPy array for feats
    batch_size = 10
    height = 6
    width = 6
    channels = 128

    feats = np.random.rand(batch_size, height, width, channels).astype(np.float32)

    # Step 2: Convert to TensorFlow tensor
    feats_tf = tf.convert_to_tensor(feats)

    # Step 3: Convert to PyTorch tensor
    feats_torch = torch.tensor(feats)

    # Step 4: Flatten the second and third dimensions (height and width) for each format
    # NumPy
    feats_np_flatten = feats.reshape(feats.shape[0], -1, feats.shape[-1])

    # TensorFlow
    feats_tf_flatten = tf.reshape(feats_tf, [feats_tf.shape[0], -1, feats_tf.shape[-1]])

    output = torch_flatten(feats_tf, 1, 2)

    print("output = ", output)

    # PyTorch
    feats_torch_flatten = feats_torch.flatten(start_dim=1, end_dim=2)

    # Step 5: Compare the results
    # Convert TensorFlow and PyTorch tensors back to NumPy arrays for comparison
    feats_tf_flatten_np = feats_tf_flatten.numpy()
    feats_torch_flatten_np = feats_torch_flatten.numpy()

    assert np.allclose(feats_np_flatten, feats_tf_flatten_np)

    assert np.allclose(feats_np_flatten, feats_torch_flatten_np)

    assert np.allclose(feats_tf_flatten_np, feats_torch_flatten_np)

    assert np.allclose(feats_np_flatten, output.numpy())

    # Check if they are equal
    print("NumPy vs TensorFlow:", np.allclose(feats_np_flatten, feats_tf_flatten_np))
    print("NumPy vs PyTorch:", np.allclose(feats_np_flatten, feats_torch_flatten_np))
    print("TensorFlow vs PyTorch:", np.allclose(feats_tf_flatten_np, feats_torch_flatten_np))

    print("numpy vs Mine:", np.allclose(feats_np_flatten, output.numpy()))



    print("Shape after flattening:")
    print("NumPy:", feats_np_flatten.shape)
    print("TensorFlow:", feats_tf_flatten.shape)
    print("PyTorch:", feats_torch_flatten.shape)



test_flatten()