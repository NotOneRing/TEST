import unittest
import numpy as np
import torch
import tensorflow as tf

from util.torch_to_tf import torch_full

class TestFull(unittest.TestCase):
    def test_full(self):
        """Test full functionality across NumPy, PyTorch, and TensorFlow."""
        # numpy_array = np.full((3, 3), 5.0)
        
        # torch_tensor = torch.tensor(numpy_array, dtype=torch.float32)
        # tf_tensor = tf.convert_to_tensor(numpy_array, dtype=tf.float32)

        torch_result = torch.full((3, 3), 5.0, dtype=torch.float32)

        tf_result = torch_full([3, 3], 5.0)

        # print("Original NumPy array:\n", numpy_array)
        # print("\nTorch tensor:\n", torch_tensor)
        # print("\nTensorFlow tensor:\n", tf_tensor)
        # print("\nTorch full tensor:\n", torch_result)
        # print("\nTensorFlow fill tensor:\n", tf_result)

        # Assert that PyTorch and TensorFlow results are close
        self.assertTrue(np.allclose(torch_result.numpy(), tf_result.numpy()))


if __name__ == '__main__':
    unittest.main()
