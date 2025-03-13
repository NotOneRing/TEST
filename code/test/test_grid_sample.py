import torch
import tensorflow as tf
import numpy as np
import unittest

from util.torch_to_tf import torch_nn_functional_grid_sample


class TestGridSample(unittest.TestCase):
    def setUp(self):
        # fix random seeds to ensure reproducibility
        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        
        # Common setup for all tests
        self.input_tensor_torch = torch.range(start=1, end=1*3*3*3).reshape(1, 3, 3, 3)
        self.input_tensor_torch = self.input_tensor_torch.permute(0, 3, 1, 2)  # Convert to NCHW format for PyTorch
        
        self.input_tensor_tf = tf.convert_to_tensor(self.input_tensor_torch.numpy())
        
        self.grid_torch = torch.range(start=1, end=1*3*3*2).reshape(1, 3, 3, 2)

    def test_case1_align_corners(self):
        # Prepare grid with specific transformation
        grid_torch = (self.grid_torch * 2 - 1*3*3*2) / (1*3*3*2)
        grid_tf = tf.convert_to_tensor(grid_torch.numpy())
        
        # Run PyTorch implementation
        output_torch = torch.nn.functional.grid_sample(self.input_tensor_torch, grid_torch, align_corners=True)
        
        # Run TensorFlow implementation
        output_tf = torch_nn_functional_grid_sample(self.input_tensor_tf, grid_tf, align_corners=True)
        
        # Convert outputs to numpy for comparison
        torch_output = output_torch.detach().numpy()
        tensorflow_output = output_tf.numpy()
        
        # Uncomment for debugging
        # print("torch_output = ", torch_output)
        # print("tensorflow_output = ", tensorflow_output)
        
        # Assert outputs are close
        self.assertTrue(np.allclose(torch_output, tensorflow_output, atol=1e-5))

    def test_case2_align_corners(self):
        # Prepare grid with specific transformation
        grid_torch = (self.grid_torch * 2 - 1*3*3*2) / (1*3*3*2)
        grid_tf = tf.convert_to_tensor(grid_torch.numpy())
        
        # Run PyTorch implementation
        output_torch = torch.nn.functional.grid_sample(self.input_tensor_torch, grid_torch, align_corners=False)
        
        # Run TensorFlow implementation
        output_tf = torch_nn_functional_grid_sample(self.input_tensor_tf, grid_tf, align_corners=False)
        
        # Convert outputs to numpy for comparison
        torch_output = output_torch.detach().numpy()
        tensorflow_output = output_tf.numpy()
        
        # Assert outputs are close
        self.assertTrue(np.allclose(torch_output, tensorflow_output, atol=1e-5))

    def test_case3(self):
        # Prepare grid with different transformation
        grid_torch = (self.grid_torch - 1*3*3*2 / 2) / (1*3*3*2)
        grid_tf = tf.convert_to_tensor(grid_torch.numpy())
        
        # Run PyTorch implementation
        output_torch = torch.nn.functional.grid_sample(self.input_tensor_torch, grid_torch, align_corners=True)
        
        # Run TensorFlow implementation
        output_tf = torch_nn_functional_grid_sample(self.input_tensor_tf, grid_tf, align_corners=True)
        
        # Convert outputs to numpy for comparison
        torch_output = output_torch.detach().numpy()
        tensorflow_output = output_tf.numpy()
        
        # Assert outputs are close
        self.assertTrue(np.allclose(torch_output, tensorflow_output, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
