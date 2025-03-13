import unittest
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn

from util.torch_to_tf import nn_Embedding


class TestEmbedding(unittest.TestCase):
    """Test case for comparing PyTorch and TensorFlow embedding implementations."""
    
    def setUp(self):
        """Set up common test parameters."""
        self.num_embeddings = 10
        self.embedding_dim = 5
        self.padding_idx = 2
        self.input_tensor = torch.tensor([1, 2, 3, 4])
    
    def torch_embedding(self):
        """Create a PyTorch embedding layer and return weights, input, and output."""
        torch_embedding = nn.Embedding(
            self.num_embeddings, 
            self.embedding_dim, 
            padding_idx=self.padding_idx
        )

        # Manually set weights for consistency
        weights = torch.randn(self.num_embeddings, self.embedding_dim)
        torch_embedding.weight.data = weights
        torch_embedding.weight.data[self.padding_idx] = 0  # Ensure padding_idx is zeroed

        output = torch_embedding(self.input_tensor)

        return weights.numpy(), self.input_tensor.numpy(), output.detach().numpy()

    def tf_embedding(self, torch_weights, torch_input):
        """Create a TensorFlow embedding layer using PyTorch weights and return output."""
        tf_embedding = nn_Embedding(
            self.num_embeddings, 
            self.embedding_dim, 
            padding_idx=self.padding_idx, 
            _weight=torch_weights
        )

        input_tensor = tf.constant(torch_input)
        output = tf_embedding(input_tensor)

        return output.numpy()

    def test_embedding_outputs_match(self):
        """Test that PyTorch and TensorFlow embedding outputs match."""
        # Get PyTorch embedding results
        torch_weights, torch_input, torch_output = self.torch_embedding()
        
        # Get TensorFlow embedding results
        tf_output = self.tf_embedding(torch_weights, torch_input)
        
        # Compare outputs with detailed error message
        self.assertTrue(
            np.allclose(torch_output, tf_output, atol=1e-5),
            f"PyTorch and TensorFlow embedding outputs do not match.\n"
            f"PyTorch output: {torch_output}\n"
            f"TensorFlow output: {tf_output}"
        )
    
    def test_padding_idx_is_zero(self):
        """Test that the padding_idx in the embedding has zero weights."""
        torch_weights, _, _ = self.torch_embedding()
        
        # Check that padding_idx row is all zeros
        padding_weights = torch_weights[self.padding_idx]
        self.assertTrue(
            np.allclose(padding_weights, np.zeros_like(padding_weights)),
            f"Padding index {self.padding_idx} weights are not zero: {padding_weights}"
        )


if __name__ == "__main__":
    unittest.main()
