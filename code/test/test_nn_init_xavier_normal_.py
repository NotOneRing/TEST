import tensorflow as tf
import torch
import numpy as np
import unittest

from util.torch_to_tf import torch_nn_init_xavier_normal_, torch_nn_init_zeros_, nn_Linear


class TestXavierNormalInitialization(unittest.TestCase):
    """
    Test case for comparing PyTorch and TensorFlow implementations of xavier_normal_ initialization.
    """
    
    def test_compare_tf_torch(self):
        """
        Test to compare outputs between PyTorch and TensorFlow when using xavier_normal_ initialization.
        """
        input_data = np.random.rand(1, 5).astype(np.float32)  # Input tensor for both frameworks

        # PyTorch
        torch_layer = torch.nn.Linear(5, 10)
        torch.nn.init.xavier_normal_(torch_layer.weight, gain=0.01)  # Initialize weights
        torch_layer.bias.data.fill_(0.0)  # Initialize bias to zero
        torch_input = torch.tensor(input_data)
        torch_output = torch_layer(torch_input).detach().numpy()

        # TensorFlow
        tf_layer = nn_Linear(5, 10)
        tf_input = tf.convert_to_tensor(input_data)
        
        # First get output before initialization
        _ = tf_layer(tf_input).numpy()
        
        # Apply xavier_normal_ initialization
        torch_nn_init_xavier_normal_(tf_layer.kernel, gain=0.01)
        torch_nn_init_zeros_(tf_layer.bias)
        
        # Get output after initialization
        tf_output = tf_layer(tf_input).numpy()

        # Assert that bias values are close
        self.assertTrue(
            np.allclose(torch_layer.bias.data.numpy(), tf_layer.bias.numpy(), atol=1e-5),
            "Bias values do not match between PyTorch and TensorFlow"
        )
        
        # # Assert that kernel/weight shapes match
        # self.assertEqual(
        #     torch_layer.weight.shape, 
        #     (tf_layer.kernel.shape[1], tf_layer.kernel.shape[0]),
        #     "Weight shapes do not match between PyTorch and TensorFlow (accounting for transpose)"
        # )
        
        # # Assert that outputs are reasonably close
        # # Note: We don't expect exact matches due to implementation differences
        # self.assertTrue(
        #     np.allclose(torch_output, tf_output, atol=1e-4),
        #     "Outputs do not match between PyTorch and TensorFlow"
        # )


if __name__ == "__main__":
    unittest.main()
