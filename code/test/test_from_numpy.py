import unittest
import numpy as np
import torch
import tensorflow as tf

from util.torch_to_tf import torch_from_numpy

class TestFromNumpy(unittest.TestCase):
    
    def test_from_numpy(self):
        numpy_array = np.full((3, 3), 5.0)
        
        torch_tensor = torch.from_numpy(numpy_array)
        tf_tensor = torch_from_numpy(numpy_array)

        self.assertTrue(np.allclose(torch_tensor.numpy(), tf_tensor.numpy()))

if __name__ == '__main__':
    unittest.main()
