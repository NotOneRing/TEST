import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import unittest

from util.torch_to_tf import nn_Sequential, nn_Linear, nn_ReLU, torch_optim_Adam, model_forward_backward_gradients


class TestAdam(unittest.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        torch.manual_seed(42)
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Define a simple model in PyTorch
        self.torch_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

        # Define the same model in TensorFlow
        self.tf_model = nn_Sequential([
            nn_Linear(10, 5),
            nn_ReLU(),
            nn_Linear(5, 1)
        ])

        # Initialize TF model
        _ = self.tf_model(tf.constant(np.random.randn(1, 10).astype(np.float32)))

        # Initialize weights in TensorFlow model to match PyTorch
        for torch_layer, tf_layer in zip(self.torch_model, self.tf_model):
            if isinstance(torch_layer, nn.Linear):
                print("tf_layer = ", tf_layer)
                tf_layer.trainable_weights[0].assign(torch_layer.weight.detach().numpy().T)  # kernel
                tf_layer.trainable_weights[1].assign(torch_layer.bias.detach().numpy())     # bias

        # Define inputs and targets
        self.inputs = np.random.rand(4, 10).astype(np.float32)
        self.targets = np.random.rand(4, 1).astype(np.float32)

        # Define optimizers
        self.torch_optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=0.01)
        self.tf_optimizer = torch_optim_Adam(self.tf_model.trainable_variables, lr=0.01)

        # Define loss functions
        self.torch_loss_fn = nn.MSELoss()
        self.tf_loss_fn = tf.keras.losses.MeanSquaredError()

    def test_torch_and_tf_adam(self):
        """Test that torch_optim_Adam behaves the same as torch.optim.Adam"""
        # Training loop
        for step in range(5):
            # PyTorch
            torch_inputs = torch.tensor(self.inputs)
            torch_targets = torch.tensor(self.targets)
            print("torch_targets.device = ", torch_targets.device)

            self.torch_optimizer.zero_grad()
            torch_outputs = self.torch_model(torch_inputs)
            torch_loss = self.torch_loss_fn(torch_outputs, torch_targets)
            torch_loss.backward()

            self.torch_optimizer.step()

            # TensorFlow
            tf_inputs = tf.convert_to_tensor(self.inputs)
            tf_targets = tf.convert_to_tensor(self.targets)

            tf_loss, tf_gradients = model_forward_backward_gradients(
                tf_inputs, tf_targets, self.tf_loss_fn, self.tf_model
            )

            self.tf_optimizer.step(tf_gradients)

            # # Print losses
            # print(f"Step {step + 1}:")
            # print(f"  PyTorch Loss: {torch_loss.item():.6f}")
            # print(f"  TensorFlow Loss: {tf_loss.numpy():.6f}")

            # Assert losses are close
            self.assertTrue(
                np.allclose(torch_loss.item(), tf_loss.numpy(), atol=1e-4),
                f"Losses differ at step {step+1}: PyTorch={torch_loss.item():.6f}, TensorFlow={tf_loss.numpy():.6f}"
            )

        # Compare final outputs
        torch_final_output = self.torch_model(torch.tensor(self.inputs)).detach().numpy()
        tf_final_output = self.tf_model(self.inputs).numpy()

        # print("\nFinal Output Comparison:")
        # print(f"  PyTorch: {torch_final_output}")
        # print(f"  TensorFlow: {tf_final_output}")
        
        # Assert final outputs are close
        self.assertTrue(
            np.allclose(torch_final_output, tf_final_output, atol=1e-4),
            "Final outputs differ between PyTorch and TensorFlow implementations"
        )


if __name__ == '__main__':
    unittest.main()
