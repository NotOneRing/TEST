import unittest
import torch
import tensorflow as tf
import numpy as np
from util.torch_to_tf import torch_quantile

class TestQuantile(unittest.TestCase):
    def test_torch_quantile_2d(self):
        """Test torch.quantile with a 2D tensor and multiple quantile values."""
        # Create a 2D tensor
        a = torch.tensor([[ 0.0795, -1.2117,  0.9765],
                          [ 1.1707,  0.6706,  0.4884]])
        
        # Define quantile values
        q = torch.tensor([0.25, 0.5, 0.75])
        
        # Apply torch.quantile
        temp = torch.quantile(a, q, dim=1, keepdim=True)
        
        # Expected shape: [3, 2, 1] - 3 quantiles, 2 rows, 1 column (keepdim=True)
        self.assertEqual(temp.shape, torch.Size([3, 2, 1]))
        
        # Expected values (based on the original print output):
        # tensor([[[-0.5661],
        #         [ 0.5795]],
        #         [[ 0.0795],
        #         [ 0.6706]],
        #         [[ 0.5280],
        #         [ 0.9206]]])
        expected_values = torch.tensor([[[-0.5661], [0.5795]],
                                        [[0.0795], [0.6706]],
                                        [[0.5280], [0.9206]]])
        self.assertTrue(torch.allclose(temp, expected_values, rtol=1e-4))


    def test_tf_torch_quantile_dimNone_scalar_keepdimTrue(self):
        """Test the TensorFlow implementation of torch_quantile with a scalar tensor."""
        # Create a 2D tensor and convert to TensorFlow
        a_torch = torch.tensor([[ 0.0795, -1.2117,  0.9765],
                                [ 1.1707,  0.6706,  0.4884]])
        a_tf = tf.convert_to_tensor(a_torch.numpy())
        
        # Define quantile values and convert to TensorFlow
        q_torch = torch.tensor([0.25])
        q_tf = tf.convert_to_tensor(q_torch.numpy())
        
        # Apply torch_quantile from util.torch_to_tf
        temp = torch_quantile(a_tf, q_tf, dim=None, keepdim=True)
        
        print("temp = ", temp)
        print("temp.shape = ", temp.shape)
        # # Expected shape: [3, 2, 1] - 3 quantiles, 2 rows, 1 column
        # self.assertEqual(temp.shape, (3, 2, 1))
        
        # Compare with PyTorch implementation
        temp_torch = torch.quantile(a_torch, q_torch, dim=None, keepdim=True)
        temp_torch_np = temp_torch.numpy()

        print("temp_torch_np = ", temp_torch_np)
        print("temp_torch_np.shape = ", temp_torch_np.shape)

        # Verify TensorFlow implementation matches PyTorch
        self.assertTrue(np.allclose(temp.numpy(), temp_torch_np, rtol=1e-4))

    def test_tf_torch_quantile_dimNone_2d_keepdimTrue(self):
        """Test the TensorFlow implementation of torch_quantile with a 2d tensor, dim=None."""
        # Create a 2D tensor and convert to TensorFlow
        a_torch = torch.tensor([[ 0.0795, -1.2117,  0.9765],
                                [ 1.1707,  0.6706,  0.4884]])
        a_tf = tf.convert_to_tensor(a_torch.numpy())
        
        # Define quantile values and convert to TensorFlow
        q_torch = torch.tensor([0.25, 0.5, 0.75])
        q_tf = tf.convert_to_tensor(q_torch.numpy())
        
        # Apply torch_quantile from util.torch_to_tf
        temp = torch_quantile(a_tf, q_tf, dim=None, keepdim=True)
        
        print("temp = ", temp)
        print("temp.shape = ", temp.shape)
        # # Expected shape: [3, 2, 1] - 3 quantiles, 2 rows, 1 column
        # self.assertEqual(temp.shape, (3, 2, 1))
        
        # Compare with PyTorch implementation
        temp_torch = torch.quantile(a_torch, q_torch, dim=None, keepdim=True)
        temp_torch_np = temp_torch.numpy()

        print("temp_torch_np = ", temp_torch_np)
        print("temp_torch_np.shape = ", temp_torch_np.shape)

        # Verify TensorFlow implementation matches PyTorch
        self.assertTrue(np.allclose(temp.numpy(), temp_torch_np, rtol=1e-4))



    def test_tf_torch_quantile_dimNone_scalar_keepdimFalse(self):
        """Test the TensorFlow implementation of torch_quantile with a scalar tensor."""
        # Create a 2D tensor and convert to TensorFlow
        a_torch = torch.tensor([[ 0.0795, -1.2117,  0.9765],
                                [ 1.1707,  0.6706,  0.4884]])
        a_tf = tf.convert_to_tensor(a_torch.numpy())
        
        # Define quantile values and convert to TensorFlow
        q_torch = torch.tensor([0.25])
        q_tf = tf.convert_to_tensor(q_torch.numpy())
        
        # Apply torch_quantile from util.torch_to_tf
        temp = torch_quantile(a_tf, q_tf, dim=None, keepdim=False)
        
        print("temp = ", temp)
        print("temp.shape = ", temp.shape)
        # # Expected shape: [3, 2, 1] - 3 quantiles, 2 rows, 1 column
        # self.assertEqual(temp.shape, (3, 2, 1))
        
        # Compare with PyTorch implementation
        temp_torch = torch.quantile(a_torch, q_torch, dim=None, keepdim=False)
        temp_torch_np = temp_torch.numpy()

        print("temp_torch_np = ", temp_torch_np)
        print("temp_torch_np.shape = ", temp_torch_np.shape)

        # Verify TensorFlow implementation matches PyTorch
        self.assertTrue(np.allclose(temp.numpy(), temp_torch_np, rtol=1e-4))

    def test_tf_torch_quantile_dimNone_2d_keepdimFalse(self):
        """Test the TensorFlow implementation of torch_quantile with a 2d tensor, dim=None."""
        # Create a 2D tensor and convert to TensorFlow
        a_torch = torch.tensor([[ 0.0795, -1.2117,  0.9765],
                                [ 1.1707,  0.6706,  0.4884]])
        a_tf = tf.convert_to_tensor(a_torch.numpy())
        
        # Define quantile values and convert to TensorFlow
        q_torch = torch.tensor([0.25, 0.5, 0.75])
        q_tf = tf.convert_to_tensor(q_torch.numpy())
        
        # Apply torch_quantile from util.torch_to_tf
        temp = torch_quantile(a_tf, q_tf, dim=None, keepdim=False)
        
        print("temp = ", temp)
        print("temp.shape = ", temp.shape)
        # # Expected shape: [3, 2, 1] - 3 quantiles, 2 rows, 1 column
        # self.assertEqual(temp.shape, (3, 2, 1))
        
        # Compare with PyTorch implementation
        temp_torch = torch.quantile(a_torch, q_torch, dim=None, keepdim=False)
        temp_torch_np = temp_torch.numpy()

        print("temp_torch_np = ", temp_torch_np)
        print("temp_torch_np.shape = ", temp_torch_np.shape)

        # Verify TensorFlow implementation matches PyTorch
        self.assertTrue(np.allclose(temp.numpy(), temp_torch_np, rtol=1e-4))



    def test_tf_torch_quantile_2d_keepdim_True(self):
        """Test the TensorFlow implementation of torch_quantile with a 2D tensor."""
        # Create a 2D tensor and convert to TensorFlow
        a_torch = torch.tensor([[ 0.0795, -1.2117,  0.9765],
                                [ 1.1707,  0.6706,  0.4884]])
        a_tf = tf.convert_to_tensor(a_torch.numpy())
        
        # Define quantile values and convert to TensorFlow
        q_torch = torch.tensor([0.25, 0.5, 0.75])
        q_tf = tf.convert_to_tensor(q_torch.numpy())
        
        # Apply torch_quantile from util.torch_to_tf
        temp = torch_quantile(a_tf, q_tf, dim=1, keepdim=True)
        
        print("temp = ", temp)
        print("temp.shape = ", temp.shape)
        # # Expected shape: [3, 2, 1] - 3 quantiles, 2 rows, 1 column
        # self.assertEqual(temp.shape, (3, 2, 1))
        
        # Compare with PyTorch implementation
        temp_torch = torch.quantile(a_torch, q_torch, dim=1, keepdim=True)
        temp_torch_np = temp_torch.numpy()

        print("temp_torch_np = ", temp_torch_np)
        print("temp_torch_np.shape = ", temp_torch_np.shape)

        # Verify TensorFlow implementation matches PyTorch
        self.assertTrue(np.allclose(temp.numpy(), temp_torch_np, rtol=1e-4))

    def test_tf_torch_quantile_scalar_keepdim_True(self):
        """Test the TensorFlow implementation of torch_quantile with a scalar tensor."""
        # Create a 2D tensor and convert to TensorFlow
        a_torch = torch.tensor([[ 0.0795, -1.2117,  0.9765],
                                [ 1.1707,  0.6706,  0.4884]])
        a_tf = tf.convert_to_tensor(a_torch.numpy())
        
        # Define quantile values and convert to TensorFlow
        q_torch = torch.tensor([0.25])
        q_tf = tf.convert_to_tensor(q_torch.numpy())
        
        # Apply torch_quantile from util.torch_to_tf
        temp = torch_quantile(a_tf, q_tf, dim=1, keepdim=True)
        
        print("temp = ", temp)
        print("temp.shape = ", temp.shape)
        # # Expected shape: [3, 2, 1] - 3 quantiles, 2 rows, 1 column
        # self.assertEqual(temp.shape, (3, 2, 1))
        
        # Compare with PyTorch implementation
        temp_torch = torch.quantile(a_torch, q_torch, dim=1, keepdim=True)
        temp_torch_np = temp_torch.numpy()

        print("temp_torch_np = ", temp_torch_np)
        print("temp_torch_np.shape = ", temp_torch_np.shape)

        # Verify TensorFlow implementation matches PyTorch
        self.assertTrue(np.allclose(temp.numpy(), temp_torch_np, rtol=1e-4))



    def test_tf_torch_quantile_2d_keepdim_False(self):
        """Test the TensorFlow implementation of torch_quantile with a 2D tensor."""
        # Create a 2D tensor and convert to TensorFlow
        a_torch = torch.tensor([[ 0.0795, -1.2117,  0.9765],
                                [ 1.1707,  0.6706,  0.4884]])
        a_tf = tf.convert_to_tensor(a_torch.numpy())
        
        # Define quantile values and convert to TensorFlow
        q_torch = torch.tensor([0.25, 0.5, 0.75])
        q_tf = tf.convert_to_tensor(q_torch.numpy())
        
        # Apply torch_quantile from util.torch_to_tf
        temp = torch_quantile(a_tf, q_tf, dim=1, keepdim=False)
        
        print("temp = ", temp)
        print("temp.shape = ", temp.shape)
        # # Expected shape: [3, 2, 1] - 3 quantiles, 2 rows, 1 column
        # self.assertEqual(temp.shape, (3, 2, 1))
        
        # Compare with PyTorch implementation
        temp_torch = torch.quantile(a_torch, q_torch, dim=1, keepdim=False)
        temp_torch_np = temp_torch.numpy()

        print("temp_torch_np = ", temp_torch_np)
        print("temp_torch_np.shape = ", temp_torch_np.shape)

        # Verify TensorFlow implementation matches PyTorch
        self.assertTrue(np.allclose(temp.numpy(), temp_torch_np, rtol=1e-4))

    def test_tf_torch_quantile_scalar_keepdim_False(self):
        """Test the TensorFlow implementation of torch_quantile with a scalar tensor."""
        # Create a 2D tensor and convert to TensorFlow
        a_torch = torch.tensor([[ 0.0795, -1.2117,  0.9765],
                                [ 1.1707,  0.6706,  0.4884]])
        a_tf = tf.convert_to_tensor(a_torch.numpy())
        
        # Define quantile values and convert to TensorFlow
        # q_torch = torch.tensor([0.25, 0.5, 0.75])
        q_torch = torch.tensor([0.25])
        q_tf = tf.convert_to_tensor(q_torch.numpy())
        
        # Apply torch_quantile from util.torch_to_tf
        temp = torch_quantile(a_tf, q_tf, dim=1, keepdim=False)
        
        print("temp = ", temp)
        print("temp.shape = ", temp.shape)
        # # Expected shape: [3, 2, 1] - 3 quantiles, 2 rows, 1 column
        # self.assertEqual(temp.shape, (3, 2, 1))
        
        # Compare with PyTorch implementation
        temp_torch = torch.quantile(a_torch, q_torch, dim=1, keepdim=False)
        temp_torch_np = temp_torch.numpy()

        print("temp_torch_np = ", temp_torch_np)
        print("temp_torch_np.shape = ", temp_torch_np.shape)

        # Verify TensorFlow implementation matches PyTorch
        self.assertTrue(np.allclose(temp.numpy(), temp_torch_np, rtol=1e-4))


    def test_torch_quantile_scalar(self):
        """Test torch.quantile with scalar values and clipping."""
        # Create a 1D tensor
        advantages = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        advantages = torch.tensor(advantages)
        
        # Define quantile values
        self_clip_advantage_lower_quantile = 0.1
        self_clip_advantage_upper_quantile = 0.9
        
        # Calculate min and max using quantile
        advantage_min = torch.quantile(advantages, self_clip_advantage_lower_quantile)
        advantage_max = torch.quantile(advantages, self_clip_advantage_upper_quantile)
        
        # Expected values
        self.assertAlmostEqual(advantage_min.item(), 1.4, delta=0.1)
        self.assertAlmostEqual(advantage_max.item(), 4.6, delta=0.1)
        
        # Test another percentile
        percent = 0.25
        advantage = torch.quantile(advantages, percent)
        self.assertAlmostEqual(advantage.item(), 2.0, delta=0.1)
        
        # Test clamp operation
        clamped_advantages = advantages.clamp(min=advantage_min, max=advantage_max)
        expected_clamped = torch.tensor([1.4, 2.0, 3.0, 4.0, 4.6], dtype=torch.float32)
        self.assertTrue(torch.allclose(clamped_advantages, expected_clamped, rtol=1e-1))

if __name__ == '__main__':
    unittest.main()
