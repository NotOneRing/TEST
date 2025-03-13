import unittest
import numpy as np

from util.torch_to_tf import torch_log, torch_tensor

class TestLog(unittest.TestCase):
    def test_log(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])

        import torch

        torch_a = torch.tensor(a)
        torch_result = torch.log(torch_a)
        # print(torch_result)

        tf_a = torch_tensor(a)
        tf_result = torch_log(tf_a)
        # print(tf_result)

        self.assertTrue(np.allclose(torch_result.numpy(), tf_result.numpy()))


if __name__ == '__main__':
    unittest.main()
