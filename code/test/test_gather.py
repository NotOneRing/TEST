import unittest
import tensorflow as tf
import torch
import copy
from util.torch_to_tf import torch_gather
from util.func_pytest import compare_result, np_to_tf, np_to_torch, gen_3d_int, gen_3d_index


class TestGather(unittest.TestCase):
    def test_gather_3d_dimensions(self):
        """Test gather operation on 3D tensors along different dimensions."""
        input_tensor = gen_3d_int()[0]
        index_tensor = gen_3d_index()[0]

        tf_input_tensor = np_to_tf(input_tensor)
        tf_index_tensor = np_to_tf(index_tensor)

        torch_input_tensor = np_to_torch(input_tensor)
        torch_index_tensor = np_to_torch(index_tensor)

        # Gather along the 0th dimension (dim == 0)
        out_tf_0 = torch_gather(tf_input_tensor, 0, tf_index_tensor)
        out_torch_0 = torch.gather(torch_input_tensor, 0, torch_index_tensor)
        self.assertTrue(compare_result(out_torch_0, out_tf_0))

        # Gather along the 1st dimension (dim == 1)
        out_tf_1 = torch.gather(torch_input_tensor, 1, torch_index_tensor)
        out_torch_1 = torch_gather(tf_input_tensor, 1, tf_index_tensor)
        self.assertTrue(compare_result(out_torch_1, out_tf_1))

        # Gather along the 2nd dimension (dim == 2)
        out_torch_2 = torch.gather(torch_input_tensor, 2, torch_index_tensor)
        out_tf_2 = torch_gather(tf_input_tensor, 2, tf_index_tensor)
        self.assertTrue(compare_result(out_torch_2, out_tf_2))

    def test_gather_with_repeat(self):
        """Test gather operation with repeated tensors."""
        a = torch.tensor(
            [1.0040205, 1.0142527, 1.0310161, 1.0548513, 1.0865637, 1.1272943,
             1.178628, 1.2427604, 1.3227621, 1.4230014, 1.5498539, 1.7129476,
             1.9274747, 2.21882, 2.632745, 3.2608492, 4.3169575, 6.442673,
             12.846252, 406.2368],
            dtype=torch.float32)

        a = a.unsqueeze(-1)
        a = a.repeat((1, 2))

        t = torch.full((18,), 1, dtype=torch.int64)
        t = t.unsqueeze(-1)
        t = t.repeat((1, 2))

        a_tf = tf.convert_to_tensor(a.numpy())
        t_tf = tf.convert_to_tensor(t.numpy())

        out = torch_gather(a_tf, -1, t_tf)
        out2 = a.gather(-1, t)

        self.assertTrue(compare_result(out, out2))

    def test_gather_with_full_tensor(self):
        """Test gather operation with full tensors."""
        a = torch.tensor(
            [1.0040205, 1.0142527, 1.0310161, 1.0548513, 1.0865637, 1.1272943,
             1.178628, 1.2427604, 1.3227621, 1.4230014, 1.5498539, 1.7129476,
             1.9274747, 2.21882, 2.632745, 3.2608492, 4.3169575, 6.442673,
             12.846252, 406.2368],
            dtype=torch.float32)

        t = torch.full((128,), 19, dtype=torch.int64)

        a_tf = tf.convert_to_tensor(a.numpy())
        t_tf = tf.convert_to_tensor(t.numpy())

        out = torch_gather(a_tf, -1, t_tf)
        out2 = a.gather(-1, t)

        self.assertTrue(compare_result(out, out2))


if __name__ == "__main__":
    unittest.main()
