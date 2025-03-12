import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import unittest

from util.torch_to_tf import nn_Linear, nn_ReLU, torch_optim_AdamW, nn_Sequential


class TestAdamW(unittest.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        torch.manual_seed(42)
        tf.random.set_seed(42)
        np.random.seed(42)
        
    
    def test_torch_and_tf_adamw(self):
        """Test that torch.optim.AdamW and our TensorFlow implementation behave similarly"""
        # Define a simple model in PyTorch
        torch_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

        # Define the same model in TensorFlow
        tf_model = nn_Sequential([
            nn_Linear(10, 5),
            nn_ReLU(),
            nn_Linear(5, 1)
        ])

        tf_model.build(input_shape=(None, 10))

        # Initialize model
        _ = tf_model(tf.constant(np.random.randn(1, 10).astype(np.float32)))

        # Define inputs and targets
        self.inputs = np.random.rand(4, 10).astype(np.float32)
        self.targets = np.random.rand(4, 1).astype(np.float32)

        # Initialize weights in TensorFlow model to match PyTorch
        for torch_layer, tf_layer in zip(torch_model, tf_model):
            if isinstance(torch_layer, nn.Linear):
                tf_layer.trainable_weights[0].assign(torch_layer.weight.detach().numpy().T)  # kernel
                tf_layer.trainable_weights[1].assign(torch_layer.bias.detach().numpy())     # bias

        # Define optimizers
        torch_optimizer = torch.optim.AdamW(torch_model.parameters(), lr=0.01, weight_decay=0.01)
        tf_optimizer = torch_optim_AdamW(tf_model.trainable_variables, lr=0.01, weight_decay=0.01)

        # Define loss functions
        torch_loss_fn = nn.MSELoss()
        tf_loss_fn = tf.keras.losses.MeanSquaredError()

        # Training loop
        for step in range(5):
            # PyTorch
            torch_inputs = torch.tensor(self.inputs)
            torch_targets = torch.tensor(self.targets)

            torch_optimizer.zero_grad()
            torch_outputs = torch_model(torch_inputs)
            torch_loss = torch_loss_fn(torch_outputs, torch_targets)
            torch_loss.backward()
            torch_optimizer.step()

            # TensorFlow
            with tf.GradientTape() as tape:
                tf_outputs = tf_model(self.inputs)
                tf_loss = tf_loss_fn(self.targets, tf_outputs)
            tf_gradients = tape.gradient(tf_loss, tf_model.trainable_variables)

            tf_optimizer.step(tf_gradients)

            # # Print losses
            # print(f"Step {step + 1}:")
            # print(f"  PyTorch Loss: {torch_loss.item():.6f}")
            # print(f"  TensorFlow Loss: {tf_loss.numpy():.6f}")

            self.assertAlmostEqual(torch_loss.item(), tf_loss.numpy(), delta=1e-4)

        # Compare final outputs
        torch_final_output = torch_model(torch.tensor(self.inputs)).detach().numpy()
        tf_final_output = tf_model(self.inputs).numpy()

        # print("\nFinal Output Comparison:")
        # print(f"  PyTorch: {torch_final_output}")
        # print(f"  TensorFlow: {tf_final_output}")
        
        # Check if outputs are close
        self.assertTrue(np.allclose(torch_final_output, tf_final_output, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
