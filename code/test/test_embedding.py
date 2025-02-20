import tensorflow as tf
import numpy as np

import torch

import torch.nn as nn

from util.torch_to_tf import nn_Embedding


# Test functions
def torch_embedding():
    num_embeddings = 10
    embedding_dim = 5
    padding_idx = 2

    torch_embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    # Manually set weights for consistency
    weights = torch.randn(num_embeddings, embedding_dim)
    torch_embedding.weight.data = weights
    torch_embedding.weight.data[padding_idx] = 0  # Ensure padding_idx is zeroed

    input_tensor = torch.tensor([1, 2, 3, 4])
    output = torch_embedding(input_tensor)

    return weights.numpy(), input_tensor.numpy(), output.detach().numpy()

def tf_embedding(torch_weights, torch_input, num_embeddings, embedding_dim, padding_idx):
    tf_embedding = nn_Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, _weight=torch_weights)

    input_tensor = tf.constant(torch_input)
    output = tf_embedding(input_tensor)

    return output.numpy()

def test_embeddings():
    # Parameters
    num_embeddings = 10
    embedding_dim = 5
    padding_idx = 2

    # Test PyTorch embedding
    torch_weights, torch_input, torch_output = torch_embedding()

    # Test TensorFlow embedding
    tf_output = tf_embedding(torch_weights, torch_input, num_embeddings, embedding_dim, padding_idx)

    # Compare outputs
    print("Torch output:", torch_output)
    print("TF output:", tf_output)

    match = np.allclose(torch_output, tf_output, atol=1e-5)
    assert match
    print(f"Output match: {match}")

if __name__ == "__main__":
    test_embeddings()




