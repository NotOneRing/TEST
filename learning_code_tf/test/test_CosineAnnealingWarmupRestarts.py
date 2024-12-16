import torch
import torch.optim as optim
import torch.nn as nn

from util.scheduler import CosineAnnealingWarmupRestarts

import math

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
scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=5, cycle_mult=2.0, max_lr=0.1, min_lr=0.001, warmup_steps=2)

pytorch_lr_list = []
# Simulate training loop and print learning rate every epoch
for epoch in range(10):
    # scheduler.step(epoch)
    scheduler.step()
    lr_epoch = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}, Learning rate: {lr_epoch}")
    pytorch_lr_list.append(lr_epoch)















print("Finish Pytorch")
print("")




import math
import tensorflow as tf
import numpy as np

from util.torch_to_tf import nn_Linear


# Define the model (same structure for both PyTorch and TensorFlow)
class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn_Linear(10, 10)
        self.fc2 = nn_Linear(10, 1)

    def call(self, inputs):
        x = self.fc1(inputs)
        return self.fc2(x)




import tensorflow as tf
import math

from util.torch_to_tf import tf_CosineAnnealingWarmupRestarts



        
# Example usage:
# Define the optimizer with the custom learning rate schedule
initial_learning_rate = 0.1
first_cycle_steps = 5
warmup_steps = 2

lr_schedule = tf_CosineAnnealingWarmupRestarts(
    first_cycle_steps=first_cycle_steps,
    cycle_mult=2.0,
    max_lr=0.1,
    min_lr=0.001,
    warmup_steps=warmup_steps,
    gamma=1.0
)


# Define a dummy optimizer (just for example)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Initialize the model (dummy model for testing)
model = SimpleModel()


tf_lr_list = []

# Simulate training loop and print learning rate every epoch
for epoch in range(10):
    # print("type(epoch) = ", type(epoch))

    # Perform one step of training (dummy data here)
    x = np.random.rand(1, 10)
    y = np.random.rand(1, 1)
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.mean_squared_error(y, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


    # lr_epoch = lr_schedule(epoch)
    # lr_epoch = lr_schedule()
    # Print learning rate using the lr_schedule directly
    print(f"Epoch {epoch+1}, Learning rate: {lr_epoch}")  # Accessing the learning rate value
    tf_lr_list.append(lr_epoch)


for i in range(10):
    print("pytorch_lr_list[{}] = {}".format(i, pytorch_lr_list[i]))
    print("tf_lr_list[{}] = {}".format(i, tf_lr_list[i]))























# import tensorflow as tf
# import torch
# import torch.nn as nn
# import numpy as np

# from util.torch_to_tf import nn_Sequential, nn_Linear, nn_ReLU, torch_optim_Adam, model_forward_backward_gradients


# # class torch_optim_Adam:
# #     def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
# #         """
# #         A TensorFlow implementation of torch.optim.Adam.

# #         Args:
# #             params (list): List of TensorFlow variables to optimize.
# #             lr (float): Learning rate.
# #             betas (tuple): Coefficients used for computing running averages of gradient and its square.
# #             eps (float): Term added to the denominator to improve numerical stability.
# #             weight_decay (float): Weight decay (L2 penalty).
# #         """
# #         self.params = params
# #         self.lr = lr
# #         self.betas = betas
# #         self.eps = eps
# #         self.weight_decay = weight_decay
        
# #         # TensorFlow Adam optimizer
# #         self.optimizer = tf.keras.optimizers.Adam(
# #             learning_rate=lr, beta_1=betas[0], beta_2=betas[1], epsilon=eps
# #         )

# #     def zero_grad(self):
# #         """No-op function for compatibility, gradients are reset automatically in TensorFlow."""
# #         pass


# #     def step(self, gradients):
# #         """Apply gradients to parameters."""
# #         self.optimizer.apply_gradients(zip(gradients, self.params))







# # Testing torch_optim_Adam vs torch.optim.Adam
# def test_torch_and_tf_adam():
#     # Set seeds for reproducibility
#     torch.manual_seed(42)
#     tf.random.set_seed(42)

#     # Define a simple model in PyTorch
#     torch_model = nn.Sequential(
#         nn.Linear(10, 5),
#         nn.ReLU(),
#         nn.Linear(5, 1)
#     )

#     # Define the same model in TensorFlow
#     tf_model = nn_Sequential([
#         # tf.keras.layers.Dense(5, activation='relu', input_shape=(10,)),
#         # tf.keras.layers.Dense(1)
#         nn_Linear(10, 5),
#         nn_ReLU(),
#         nn_Linear(5, 1)
#     ])

#     # tf_model.build( input_shape = (None, 10) )
#     _ = tf_model(tf.constant(np.random.randn(1, 10).astype(np.float32)))

#     # Initialize weights in TensorFlow model to match PyTorch
#     for torch_layer, tf_layer in zip(torch_model, tf_model):
#         if isinstance(torch_layer, nn.Linear):
#             # tf_layer.model.kernel.assign(tf.convert_to_tensor(torch_layer.weight.data.numpy().T, dtype=tf.float32))
#             # tf_layer.model.bias.assign(tf.convert_to_tensor(torch_layer.bias.data.numpy(), dtype=tf.float32))
#             print("tf_layer = ", tf_layer)

#             tf_layer.trainable_weights[0].assign(torch_layer.weight.detach().numpy().T)  # kernel
#             tf_layer.trainable_weights[1].assign(torch_layer.bias.detach().numpy())     # bias



#     # Define inputs and targets
#     inputs = np.random.rand(4, 10).astype(np.float32)
#     targets = np.random.rand(4, 1).astype(np.float32)

#     # Define optimizers
#     torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)

#     from util.scheduler import CosineAnnealingWarmupRestarts
#     # Initialize the scheduler
#     scheduler = CosineAnnealingWarmupRestarts(torch_optimizer, first_cycle_steps=5, cycle_mult=2.0, max_lr=0.1, min_lr=0.001, warmup_steps=2)


#     tf_optimizer = torch_optim_Adam(tf_model.trainable_variables, lr=0.01)

#     # Define loss functions
#     torch_loss_fn = nn.MSELoss()
#     tf_loss_fn = tf.keras.losses.MeanSquaredError()

#     # Training loop
#     for step in range(5):
#         # PyTorch
#         torch_inputs = torch.tensor(inputs)
#         torch_targets = torch.tensor(targets)

#         torch_optimizer.zero_grad()
#         torch_outputs = torch_model(torch_inputs)
#         torch_loss = torch_loss_fn(torch_outputs, torch_targets)
#         torch_loss.backward()

#         torch_optimizer.step()
#         scheduler.step()


#         tf_loss, tf_gradients = model_forward_backward_gradients(inputs, targets, tf_loss_fn, tf_model)
#         # # TensorFlow
#         # with tf.GradientTape() as tape:
#         #     tf_outputs = tf_model(inputs)
#         #     tf_loss = tf_loss_fn(targets, tf_outputs)
#         # tf_gradients = tape.gradient(tf_loss, tf_model.trainable_variables)

#         tf_optimizer.step(tf_gradients)



#         # Print losses
#         print(f"Step {step + 1}:")
#         print(f"  PyTorch Loss: {torch_loss.item():.6f}")
#         print(f"  TensorFlow Loss: {tf_loss.numpy():.6f}")

#         # print("np.allclose(torch_loss.item(), tf_loss.numpy()) = ", np.allclose(torch_loss.item(), tf_loss.numpy()) )
#         assert np.allclose(torch_loss.item(), tf_loss.numpy(), atol = 1e-6)

#     # Compare final outputs
#     torch_final_output = torch_model(torch.tensor(inputs)).detach().numpy()
#     tf_final_output = tf_model(inputs).numpy()

#     print("\nFinal Output Comparison:")
#     print(f"  PyTorch: {torch_final_output}")
#     print(f"  TensorFlow: {tf_final_output}")
#     # print("np.allclose(torch_final_output, tf_final_output) = ", np.allclose(torch_final_output, tf_final_output) )
#     assert np.allclose(torch_final_output, tf_final_output)





# # Run the test
# test_torch_and_tf_adam()


