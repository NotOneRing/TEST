import unittest
import torch
import tensorflow as tf

from util.torch_to_tf import torch_arange

class TestRange(unittest.TestCase):
    def test_torch_arange(self):
        # torch.arange
        tensor = torch.arange(start=0, end=10, step=2, dtype=torch.float32)
        # Check if tensor has the expected values
        expected = [0., 2., 4., 6., 8.]
        self.assertEqual(tensor.tolist(), expected)
        # Original print: tensor([0., 2., 4., 6., 8.])

    def test_tf_arange(self):
        # def torch_arange(start, end, step, dtype):
        #     return tf.range(start=start, limit=end, delta=step, dtype=dtype)

        # tf.range
        tensor = torch_arange(start=0, end=10, step=2, dtype=tf.float32)
        # Check if tensor has the expected values
        expected = [0., 2., 4., 6., 8.]
        self.assertEqual(tensor.numpy().tolist(), expected)
        # Original print: tf.Tensor([0. 2. 4. 6. 8.], shape=(5,), dtype=float32)

if __name__ == '__main__':
    unittest.main()
