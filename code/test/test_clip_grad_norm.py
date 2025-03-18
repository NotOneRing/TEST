import unittest
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from util.torch_to_tf import torch_optim_Adam, nn_Linear, nn_ReLU, nn_Sequential, model_forward_backward_gradients, torch_nn_utils_clip_grad_norm_and_step


class TestClipGradNorm(unittest.TestCase):
    def setUp(self):
        """Set up the test environment before each test method."""
        # Set random seeds to ensure reproducibility
        torch.manual_seed(0)
        np.random.seed(0)
        tf.random.set_seed(0)
        
        # Create TensorFlow model
        self.tf_model = nn_Sequential([
            nn_Linear(5, 10),
            nn_ReLU(),
            nn_Linear(10, 1)
        ])
        
        # Create PyTorch model
        class PyTorchModel(nn.Module):
            def __init__(self):
                super(PyTorchModel, self).__init__()
                self.linear1 = nn.Linear(5, 10)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(10, 1)

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x
        
        self.torch_model = PyTorchModel()
        
        # Initialize TensorFlow model
        self.tf_model.build(input_shape=(None, 5))
        _ = self.tf_model(tf.constant(np.random.randn(1, 5).astype(np.float32)))
        
        # Set the same initialization in TensorFlow and PyTorch
        with torch.no_grad():
            self.torch_model.linear1.weight.copy_(torch.from_numpy(self.tf_model[0].model.kernel.numpy().T))
            self.torch_model.linear1.bias.copy_(torch.from_numpy(self.tf_model[0].model.bias.numpy()))
            self.torch_model.linear2.weight.copy_(torch.from_numpy(self.tf_model[2].model.kernel.numpy().T))
            self.torch_model.linear2.bias.copy_(torch.from_numpy(self.tf_model[2].model.bias.numpy()))
        
        # Define optimizers
        self.tf_optimizer = torch_optim_Adam(self.tf_model.trainable_variables, lr=0.01)
        self.torch_optimizer = optim.Adam(self.torch_model.parameters(), lr=0.01)
        
        # Create input data (keep consistency)
        self.tf_x = tf.random.normal([32, 5])
        self.tf_y = tf.random.normal([32, 1])
        self.torch_x = torch.tensor(self.tf_x.numpy(), dtype=torch.float32)
        self.torch_y = torch.tensor(self.tf_y.numpy(), dtype=torch.float32)
        
        # Define loss function for TensorFlow
        self.tf_loss_fn = lambda x, y: tf.reduce_mean(tf.square(x - y))

    def test_single_step_gradient_clipping(self):
        """Test gradient clipping for a single training step."""
        # PyTorch: train and clip gradients
        self.torch_optimizer.zero_grad()
        torch_output = self.torch_model(self.torch_x)
        torch_loss = torch.nn.functional.mse_loss(torch_output, self.torch_y)
        torch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.torch_model.parameters(), max_norm=1.0)
        
        clipped_grad = torch.stack([torch.norm(p.grad) for p in self.torch_model.parameters() if p.grad is not None])
        total_norm = torch.norm(clipped_grad)
        
        self.torch_optimizer.step()
        
        # TensorFlow: train and clip gradients
        tf_loss, tf_gradients = model_forward_backward_gradients(
            self.tf_x, self.tf_y, self.tf_loss_fn, self.tf_model
        )
        
        clipped_tf_grads = torch_nn_utils_clip_grad_norm_and_step(
            self.tf_model.trainable_variables, self.tf_optimizer, max_norm=1.0, grads=tf_gradients
        )
        
        stacked_tf_clipped_grad = tf.stack([tf.norm(g) for g in clipped_tf_grads if g is not None])
        clipped_tf_grad_norm = tf.norm(stacked_tf_clipped_grad)
        
        # print("stacked_tf_clipped_grad = ", stacked_tf_clipped_grad)
        # print("clipped_grad = ", clipped_grad)
        
        # Assert that the clipped gradients are close
        self.assertTrue(
            np.allclose(stacked_tf_clipped_grad.numpy(), clipped_grad.numpy(), atol=1e-4),
            "Clipped gradients should be the same for PyTorch and TensorFlow"
        )
        
        # Assert that the losses are close
        self.assertTrue(
            np.allclose(torch_loss.item(), tf_loss.numpy(), atol=1e-4),
            "Losses should be the same for PyTorch and TensorFlow"
        )

    def test_multiple_step_training(self):
        """Test multiple training steps with gradient clipping."""
        for step in range(5):
            # PyTorch: train and clip gradients
            self.torch_optimizer.zero_grad()
            torch_output = self.torch_model(self.torch_x)
            torch_loss = torch.nn.functional.mse_loss(torch_output, self.torch_y)
            torch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.torch_model.parameters(), max_norm=1.0)
            
            clipped_grad = torch.stack([torch.norm(p.grad) for p in self.torch_model.parameters() if p.grad is not None])
            total_norm = torch.norm(clipped_grad)
            
            self.torch_optimizer.step()
            
            # TensorFlow: train and clip gradients
            tf_loss, tf_gradients = model_forward_backward_gradients(
                self.tf_x, self.tf_y, self.tf_loss_fn, self.tf_model
            )
            
            clipped_tf_grads = torch_nn_utils_clip_grad_norm_and_step(
                self.tf_model.trainable_variables, self.tf_optimizer, max_norm=1.0, grads=tf_gradients
            )
            
            stacked_tf_clipped_grad = tf.stack([tf.norm(g) for g in clipped_tf_grads if g is not None])
            clipped_tf_grad_norm = tf.norm(stacked_tf_clipped_grad)
            
            # Assert that the clipped gradients are close
            self.assertTrue(
                np.allclose(stacked_tf_clipped_grad.numpy(), clipped_grad.numpy(), atol=1e-4),
                f"Step {step+1}: Clipped gradients should be the same for PyTorch and TensorFlow"
            )
            
            # Assert that the losses are close
            self.assertTrue(
                np.allclose(torch_loss.item(), tf_loss.numpy(), atol=1e-4),
                f"Step {step+1}: Losses should be the same for PyTorch and TensorFlow"
            )

    def test_final_model_outputs(self):
        """Test that the final model outputs are close after training."""
        # First train both models for 5 steps
        for _ in range(5):
            # PyTorch training step
            self.torch_optimizer.zero_grad()
            torch_output = self.torch_model(self.torch_x)
            torch_loss = torch.nn.functional.mse_loss(torch_output, self.torch_y)
            torch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.torch_model.parameters(), max_norm=1.0)
            self.torch_optimizer.step()
            
            # TensorFlow training step
            tf_loss, tf_gradients = model_forward_backward_gradients(
                self.tf_x, self.tf_y, self.tf_loss_fn, self.tf_model
            )
            torch_nn_utils_clip_grad_norm_and_step(
                self.tf_model.trainable_variables, self.tf_optimizer, max_norm=1.0, grads=tf_gradients
            )
        
        # Compare final outputs
        torch_final_output = self.torch_model(torch.tensor(self.tf_x.numpy())).detach().numpy()
        tf_final_output = self.tf_model(self.tf_x).numpy()

        # print("torch_final_output = ", torch_final_output)
        # print("tf_final_output = ", tf_final_output)
        
        # Assert that the final outputs are close
        self.assertTrue(
            np.allclose(torch_final_output, tf_final_output, atol=1e-3),
            "Final model outputs should be close for PyTorch and TensorFlow"
        )


if __name__ == '__main__':
    unittest.main()
