import unittest
import torch
import tensorflow as tf
import numpy as np
from util.torch_to_tf import torch_softmax

class TestSoftmax(unittest.TestCase):
    """
    Test case for comparing torch.softmax and its TensorFlow implementation.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create input data as a numpy array
        self.input = np.array(range(1, 10), dtype=np.float32).reshape(3, 3)
        # Convert input to torch tensor
        self.torch_input = torch.tensor(self.input)
        # Convert input to tensorflow tensor
        self.tf_input = tf.convert_to_tensor(self.input)
    
    def test_softmax_equivalence(self):
        """
        Test that torch_softmax function produces the same result as torch.softmax.
        """
        # Apply torch.softmax to the torch tensor
        torch_result = torch.softmax(self.torch_input, dim=1)
        
        # Apply the torch_softmax function from util.torch_to_tf
        tf_result = torch_softmax(self.torch_input, dim=1)
        
        # Convert results to numpy for comparison
        torch_result_np = torch_result.detach().numpy()
        tf_result_np = tf_result.numpy()
        
        # Assert that the results are approximately equal
        np.testing.assert_allclose(torch_result_np, tf_result_np, rtol=1e-5, atol=1e-5)
        self.assertTrue( np.allclose(torch_result_np, tf_result_np, atol=1e-5) )        

        # # Print results for debugging purposes
        # print("torch_result = ", torch_result)
        # print("tf_result = ", tf_result)
        # print("torch_result_np = ", torch_result_np)
        # print("tf_result_np = ", tf_result_np)

if __name__ == '__main__':
    unittest.main()
