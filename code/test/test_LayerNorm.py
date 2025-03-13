import unittest
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from util.torch_to_tf import nn_LayerNorm


class TestLayerNorm(unittest.TestCase):
    def test_nlp_example(self):
        """Test LayerNorm on a 3D tensor (batch, sentence_length, embedding_dim)."""
        # NLP Example
        batch, sentence_length, embedding_dim = 20, 5, 10
        embedding = torch.randn(batch, sentence_length, embedding_dim)
        
        # PyTorch implementation
        layer_norm = nn.LayerNorm(embedding_dim)
        result_torch = layer_norm(embedding)
        
        # TensorFlow implementation
        embedding_numpy = embedding.numpy()
        embedding_tf = tf.convert_to_tensor(embedding_numpy)
        layer_norm_tf = nn_LayerNorm(embedding_dim)
        result_tf = layer_norm_tf(embedding_tf)
        
        # Compare results
        diff = result_torch.detach().numpy() - result_tf.numpy()
        self.assertTrue(np.allclose(result_torch.detach().numpy(), result_tf.numpy(), atol=1e-5),
                        f"NLP example failed with difference: {diff}")

    def test_image_example(self):
        """Test LayerNorm on a 4D tensor (N, C, H, W)."""
        # Image Example
        N, C, H, W = 2, 3, 4, 5
        input_tensor = torch.randn(N, C, H, W)
        
        # PyTorch implementation
        # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        layer_norm = nn.LayerNorm([C, H, W])
        output_torch = layer_norm(input_tensor)
        
        # TensorFlow implementation
        input_tf = tf.convert_to_tensor(input_tensor.numpy())
        layer_norm_tf = nn_LayerNorm([C, H, W])
        output_tf = layer_norm_tf(input_tf)
        
        # Compare results
        diff = output_torch.detach().numpy() - output_tf.numpy()
        self.assertTrue(np.allclose(output_torch.detach().numpy(), output_tf.numpy(), atol=1e-5),
                        f"Image example failed with difference: {diff}")


if __name__ == '__main__':
    unittest.main()
