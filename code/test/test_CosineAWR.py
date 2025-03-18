import unittest
import torch
import torch.optim as optim
import torch.nn as nn
import tensorflow as tf
import numpy as np

from util.scheduler import CosineAnnealingWarmupRestarts
from util.torch_to_tf import (
    CosineAWR, 
    nn_Linear, 
    nn_Sequential, 
    nn_ReLU, 
    torch_optim_Adam, 
    torch_optim_AdamW,
    model_forward_backward_gradients
)

import tensorflow as tf

# torch.manual_seed(42)
# tf.random.set_seed(42)


class TestCosineAWR(unittest.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        torch.manual_seed(42)
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def test_learning_rate(self):
        """Test basic functionality of CosineAnnealingWarmupRestarts scheduler."""
        # Define the model (same structure for both PyTorch and TensorFlow)
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(10, 10)
                self.fc2 = nn.Linear(10, 1)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        # Initialize model and optimizer
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.1)

        # Initialize the scheduler
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            first_cycle_steps=5, 
            cycle_mult=2.0, 
            max_lr=0.1, 
            min_lr=0.001, 
            warmup_steps=2
        )

        pytorch_lr_list = []
        # Simulate training loop and collect learning rates
        for epoch in range(10):
            scheduler.step()
            lr_epoch = optimizer.param_groups[0]['lr']
            pytorch_lr_list.append(lr_epoch)

        # TensorFlow implementation
        class SimpleModelTF(tf.keras.Model):
            def __init__(self):
                super(SimpleModelTF, self).__init__()
                self.fc1 = nn_Linear(10, 10)
                self.fc2 = nn_Linear(10, 1)

            def call(self, inputs):
                x = self.fc1(inputs)
                return self.fc2(x)

        # Initialize TF scheduler
        lr_schedule = CosineAWR(
            first_cycle_steps=5,
            cycle_mult=2.0,
            max_lr=0.1,
            min_lr=0.001,
            warmup_steps=2,
            gamma=1.0
        )
 
        # Initialize the model
        model_tf = SimpleModelTF()
        optimizer_tf = torch_optim_Adam(model_tf.trainable_variables, lr=lr_schedule)

        tf_lr_list = []
        # Simulate training loop and collect learning rates
        for epoch in range(10):
            # Perform one step of training with dummy data
            x = np.random.rand(1, 10)
            y = np.random.rand(1, 1)

            x_tf = tf.convert_to_tensor(x)
            y_tf = tf.convert_to_tensor(y)

            with tf.GradientTape() as tape:
                predictions = model_tf(x_tf, training=True)
                # loss = tf.keras.losses.MeanSquaredError(y_tf, predictions)
                mse_loss = tf.keras.losses.MeanSquaredError()
                loss = mse_loss(y_tf, predictions)

            grads = tape.gradient(loss, model_tf.trainable_variables)
            optimizer_tf.apply_gradients(zip(grads, model_tf.trainable_variables))

            lr_epoch = lr_schedule.step()
            tf_lr_list.append(lr_epoch)

        # Convert to numpy arrays for comparison
        pytorch_result_arr_np = np.array(pytorch_lr_list)
        tf_result_arr_np = np.array(tf_lr_list)

        # print("pytorch_result_arr_np = ", pytorch_result_arr_np)
        # print("tf_result_arr_np = ", tf_result_arr_np)

        # Assert that PyTorch and TensorFlow learning rates are close
        self.assertTrue(np.allclose(pytorch_result_arr_np, tf_result_arr_np))



    def test_torch_and_tf_learning_rate_model(self):
        """Test CosineAWR with Adam optimizer in both PyTorch and TensorFlow."""
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

        # Initialize TF model
        _ = tf_model(tf.constant(np.random.randn(1, 10).astype(np.float32)))

        # Initialize weights in TensorFlow model to match PyTorch
        for torch_layer, tf_layer in zip(torch_model, tf_model):
            if isinstance(torch_layer, nn.Linear):
                tf_layer.trainable_weights[0].assign(torch_layer.weight.detach().numpy().T)  # kernel
                tf_layer.trainable_weights[1].assign(torch_layer.bias.detach().numpy())     # bias

        # Define inputs and targets
        inputs = np.random.rand(4, 10).astype(np.float32)
        targets = np.random.rand(4, 1).astype(np.float32)

        # Define PyTorch optimizer and scheduler
        torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)
        scheduler = CosineAnnealingWarmupRestarts(
            torch_optimizer, 
            first_cycle_steps=5, 
            cycle_mult=2.0, 
            max_lr=0.1, 
            min_lr=0.001, 
            warmup_steps=2
        )

        # Define TensorFlow scheduler and optimizer
        lr_schedule = CosineAWR(
            first_cycle_steps=5,
            cycle_mult=2.0,
            max_lr=0.1,
            min_lr=0.001,
            warmup_steps=2,
            gamma=1.0
        )
        tf_optimizer = torch_optim_Adam(tf_model.trainable_variables, lr=lr_schedule)

        # Define loss functions
        torch_loss_fn = nn.MSELoss()
        tf_loss_fn = tf.keras.losses.MeanSquaredError()

        # Training loop
        for out_step in range(3):
            for step in range(5):
                # PyTorch forward and backward pass
                torch_inputs = torch.tensor(inputs)
                torch_targets = torch.tensor(targets)

                torch_optimizer.zero_grad()
                torch_outputs = torch_model(torch_inputs)
                torch_loss = torch_loss_fn(torch_outputs, torch_targets)
                torch_loss.backward()
                torch_optimizer.step()

                # TensorFlow forward and backward pass
                tf_loss, tf_gradients = model_forward_backward_gradients(inputs, targets, tf_loss_fn, tf_model)
                tf_optimizer.step(tf_gradients)

                # print("torch_loss = ", torch_loss)
                # print("tf_loss = ", tf_loss)

                # Assert losses are close
                self.assertTrue(np.allclose(torch_loss.item(), tf_loss.numpy(), atol=1e-3))

            # Step schedulers
            scheduler.step()
            lr_schedule.step()

        # Compare final outputs
        torch_final_output = torch_model(torch.tensor(inputs)).detach().numpy()
        tf_final_output = tf_model(inputs).numpy()

        # print("torch_final_output = ", torch_final_output)
        # print("tf_final_output = ", tf_final_output)

        # Assert outputs are close
        self.assertTrue(np.allclose(torch_final_output, tf_final_output, atol=1e-3))






    def test_torch_and_tf_learning_rate_model_AdamW(self):
        """Test CosineAWR with AdamW optimizer in both PyTorch and TensorFlow."""
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

        # Initialize TF model
        _ = tf_model(tf.constant(np.random.randn(1, 10).astype(np.float32)))

        # Initialize weights in TensorFlow model to match PyTorch
        for torch_layer, tf_layer in zip(torch_model, tf_model):
            if isinstance(torch_layer, nn.Linear):
                tf_layer.trainable_weights[0].assign(torch_layer.weight.detach().numpy().T)  # kernel
                tf_layer.trainable_weights[1].assign(torch_layer.bias.detach().numpy())     # bias

        # Define inputs and targets
        inputs = np.random.rand(4, 10).astype(np.float32)
        targets = np.random.rand(4, 1).astype(np.float32)

        # Define PyTorch optimizer and scheduler
        torch_optimizer = torch.optim.AdamW(torch_model.parameters(), lr=0.01)
        scheduler = CosineAnnealingWarmupRestarts(
            torch_optimizer, 
            first_cycle_steps=5, 
            cycle_mult=2.0, 
            max_lr=0.1, 
            min_lr=0.001, 
            warmup_steps=2
        )

        # Define TensorFlow scheduler and optimizer
        lr_schedule = CosineAWR(
            first_cycle_steps=5,
            cycle_mult=2.0,
            max_lr=0.1,
            min_lr=0.001,
            warmup_steps=2,
            gamma=1.0
        )
        tf_optimizer = torch_optim_AdamW(tf_model.trainable_variables, lr=lr_schedule)

        # Define loss functions
        torch_loss_fn = nn.MSELoss()
        tf_loss_fn = tf.keras.losses.MeanSquaredError()

        # Training loop
        for out_step in range(3):
            for step in range(5):
                # PyTorch forward and backward pass
                torch_inputs = torch.tensor(inputs)
                torch_targets = torch.tensor(targets)

                torch_optimizer.zero_grad()
                torch_outputs = torch_model(torch_inputs)
                torch_loss = torch_loss_fn(torch_outputs, torch_targets)
                torch_loss.backward()
                torch_optimizer.step()

                # TensorFlow forward and backward pass
                tf_loss, tf_gradients = model_forward_backward_gradients(inputs, targets, tf_loss_fn, tf_model)
                tf_optimizer.apply_gradients(zip(tf_gradients, tf_model.trainable_variables))

                # Assert losses are close
                self.assertTrue(np.allclose(torch_loss.item(), tf_loss.numpy(), atol=1e-3))

            # Step schedulers
            scheduler.step()
            lr_schedule.step()

        # Compare final outputs
        torch_final_output = torch_model(torch.tensor(inputs)).detach().numpy()
        tf_final_output = tf_model(inputs).numpy()

        # print("torch_final_output = ", torch_final_output)
        # print("tf_final_output = ", tf_final_output)

        # Assert outputs are close
        self.assertTrue(np.allclose(torch_final_output, tf_final_output, atol=1e-3))






    def test_torch_and_tf_learning_rate_model_AdamW_2(self):
        """Test CosineAWR with AdamW optimizer in both PyTorch and TensorFlow, Tensorflow use step rather than apply_gradients."""
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

        # Initialize TF model
        _ = tf_model(tf.constant(np.random.randn(1, 10).astype(np.float32)))

        # Initialize weights in TensorFlow model to match PyTorch
        for torch_layer, tf_layer in zip(torch_model, tf_model):
            if isinstance(torch_layer, nn.Linear):
                tf_layer.trainable_weights[0].assign(torch_layer.weight.detach().numpy().T)  # kernel
                tf_layer.trainable_weights[1].assign(torch_layer.bias.detach().numpy())     # bias

        # Define inputs and targets
        inputs = np.random.rand(4, 10).astype(np.float32)
        targets = np.random.rand(4, 1).astype(np.float32)

        # Define PyTorch optimizer and scheduler
        torch_optimizer = torch.optim.AdamW(torch_model.parameters(), lr=0.01)
        scheduler = CosineAnnealingWarmupRestarts(
            torch_optimizer, 
            first_cycle_steps=5, 
            cycle_mult=2.0, 
            max_lr=0.1, 
            min_lr=0.001, 
            warmup_steps=2
        )

        # Define TensorFlow scheduler and optimizer
        lr_schedule = CosineAWR(
            first_cycle_steps=5,
            cycle_mult=2.0,
            max_lr=0.1,
            min_lr=0.001,
            warmup_steps=2,
            gamma=1.0
        )
        tf_optimizer = torch_optim_AdamW(tf_model.trainable_variables, lr=lr_schedule)

        # Define loss functions
        torch_loss_fn = nn.MSELoss()
        tf_loss_fn = tf.keras.losses.MeanSquaredError()

        # Training loop
        for out_step in range(3):
            for step in range(5):
                # PyTorch forward and backward pass
                torch_inputs = torch.tensor(inputs)
                torch_targets = torch.tensor(targets)

                torch_optimizer.zero_grad()
                torch_outputs = torch_model(torch_inputs)
                torch_loss = torch_loss_fn(torch_outputs, torch_targets)
                torch_loss.backward()
                torch_optimizer.step()

                # TensorFlow forward and backward pass
                tf_loss, tf_gradients = model_forward_backward_gradients(inputs, targets, tf_loss_fn, tf_model)
                
                # tf_optimizer.apply_gradients(zip(tf_gradients, tf_model.trainable_variables))
                tf_optimizer.step(tf_gradients)
                
                # Assert losses are close
                self.assertTrue(np.allclose(torch_loss.item(), tf_loss.numpy(), atol=1e-3))

            # Step schedulers
            scheduler.step()
            lr_schedule.step()

        # Compare final outputs
        torch_final_output = torch_model(torch.tensor(inputs)).detach().numpy()
        tf_final_output = tf_model(inputs).numpy()

        # print("torch_final_output = ", torch_final_output)
        # print("tf_final_output = ", tf_final_output)

        # Assert outputs are close
        self.assertTrue(np.allclose(torch_final_output, tf_final_output, atol=1e-3))


if __name__ == '__main__':
    unittest.main()



