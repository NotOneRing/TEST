import torch
import torch.nn as nn

from util.torch_to_tf import nn_LayerNorm

import numpy as np

def test_LayerNorm():

    # NLP Example
    batch, sentence_length, embedding_dim = 20, 5, 10

    embedding = torch.randn(batch, sentence_length, embedding_dim)

    layer_norm = nn.LayerNorm(embedding_dim)
    # Activate module
    result1 = layer_norm(embedding)
    # Image Example
    # N, C, H, W = 20, 5, 10, 10
    N, C, H, W = 2, 3, 4, 5

    input = torch.randn(N, C, H, W)
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    layer_norm = nn.LayerNorm([C, H, W])
    output1 = layer_norm(input)

    # print("output = ", output1)

    import tensorflow as tf

    embedding_numpy = embedding.numpy()

    embedding = tf.convert_to_tensor(embedding_numpy)

    layer_norm = nn_LayerNorm(embedding_dim)
    # Activate module
    tf_result1 = layer_norm(embedding)
    # Image Example
    # N, C, H, W = 20, 5, 10, 10
    input = tf.convert_to_tensor(input.numpy())
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    layer_norm = nn_LayerNorm([C, H, W])
    output2 = layer_norm(input)

    # print("input = ", input)
    # print("output2 = ", output2)


    print(result1.detach().numpy() - tf_result1.numpy())

    print(output1.detach().numpy() - output2.numpy())


    print("result = ", np.allclose(output1.detach().numpy(), output2.numpy()) )

    assert np.allclose(result1.detach().numpy(), tf_result1.numpy(), atol=1e-5)

    assert np.allclose(output1.detach().numpy(), output2.numpy(), atol=1e-5)


test_LayerNorm()

