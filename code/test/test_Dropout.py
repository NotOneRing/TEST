import tensorflow as tf
import torch
import numpy as np
import unittest

from util.torch_to_tf import nn_Dropout


class TestDropout(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility
        # Uncomment if needed for deterministic testing
        # torch.manual_seed(42)
        # tf.random.set_seed(42)
        
        # Initialize parameters
        self.input_tensor = np.random.rand(4, 4).astype(np.float32)
        self.dropout_prob = 0.5  # dropout rate

    def test_dropout_implementation(self):
        """Test that PyTorch and TensorFlow dropout implementations behave similarly"""
        # PyTorch part
        torch_dropout = torch.nn.Dropout(p=self.dropout_prob)
        torch_input = torch.tensor(self.input_tensor, requires_grad=True)
        torch_output = torch_dropout(torch_input)

        # TensorFlow part
        tf_dropout = nn_Dropout(p=self.dropout_prob)
        tf_input = tf.convert_to_tensor(self.input_tensor)
        tf_output = tf_dropout(tf_input, training=True)

        # Get numpy arrays for comparison
        torch_output_np = torch_output.detach().numpy()
        tf_output_np = tf_output.numpy()

        # Check if it satisfies dropout's behavior: the ratio of zeros should be close to dropout_prob
        torch_zeros_ratio = np.mean(torch_output_np == 0)
        tf_zeros_ratio = np.mean(tf_output_np == 0)
        
        # # Print for debugging purposes
        # print(f"Input Tensor:\n{self.input_tensor}")
        # print(f"\nPyTorch Output:\n{torch_output_np}")
        # print(f"\nTensorFlow Output:\n{tf_output_np}")
        # print(f"\nPyTorch zeros ratio: {torch_zeros_ratio:.2f}")
        # print(f"TensorFlow zeros ratio: {tf_zeros_ratio:.2f}")
        
        # Assert that zero ratios are close to the dropout probability
        # Allow for some deviation due to randomness
        self.assertAlmostEqual(torch_zeros_ratio, self.dropout_prob, delta=0.3)
        self.assertAlmostEqual(tf_zeros_ratio, self.dropout_prob, delta=0.3)
        
        # Check that non-zero values are scaled correctly (by 1/(1-p))
        scale_factor = 1.0 / (1.0 - self.dropout_prob)
        
        # Get masks (where output is non-zero)
        torch_mask = torch_output_np != 0
        tf_mask = tf_output_np != 0
        
        if np.any(torch_mask):
            # Check if PyTorch non-zero values are scaled correctly
            torch_ratio = torch_output_np[torch_mask] / self.input_tensor[torch_mask]
            # print("torch_ratio = ", torch_ratio)
            self.assertTrue(np.allclose(torch_ratio, scale_factor, rtol=1e-5))
            
        if np.any(tf_mask):
            # Check if TensorFlow non-zero values are scaled correctly
            tf_ratio = tf_output_np[tf_mask] / self.input_tensor[tf_mask]
            # print("tf_ratio = ", tf_ratio)
            self.assertTrue(np.allclose(tf_ratio, scale_factor, rtol=1e-5))

    def test_dropout_inference_mode(self):
        """Test that dropout doesn't drop values in inference mode"""
        # PyTorch part - eval mode
        torch_dropout = torch.nn.Dropout(p=self.dropout_prob)
        torch_dropout.eval()  # Set to evaluation mode
        torch_input = torch.tensor(self.input_tensor, requires_grad=True)
        torch_output = torch_dropout(torch_input)

        # TensorFlow part - training=False
        tf_dropout = nn_Dropout(p=self.dropout_prob)
        tf_input = tf.convert_to_tensor(self.input_tensor)
        tf_output = tf_dropout(tf_input, training=False)

        # Get numpy arrays for comparison
        torch_output_np = torch_output.detach().numpy()
        tf_output_np = tf_output.numpy()
        
        # In inference mode, dropout should not drop any values
        # and the output should be identical to the input
        self.assertTrue(np.allclose(torch_output_np, self.input_tensor))
        self.assertTrue(np.allclose(tf_output_np, self.input_tensor))


if __name__ == "__main__":
    unittest.main()
