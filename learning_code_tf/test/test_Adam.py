import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np

from util.torch_to_tf import nn_Sequential, nn_Linear, nn_ReLU, torch_optim_Adam, model_forward_backward_gradients


# class torch_optim_Adam:
#     def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
#         """
#         A TensorFlow implementation of torch.optim.Adam.

#         Args:
#             params (list): List of TensorFlow variables to optimize.
#             lr (float): Learning rate.
#             betas (tuple): Coefficients used for computing running averages of gradient and its square.
#             eps (float): Term added to the denominator to improve numerical stability.
#             weight_decay (float): Weight decay (L2 penalty).
#         """
#         self.params = params
#         self.lr = lr
#         self.betas = betas
#         self.eps = eps
#         self.weight_decay = weight_decay
        
#         # TensorFlow Adam optimizer
#         self.optimizer = tf.keras.optimizers.Adam(
#             learning_rate=lr, beta_1=betas[0], beta_2=betas[1], epsilon=eps
#         )

#     def zero_grad(self):
#         """No-op function for compatibility, gradients are reset automatically in TensorFlow."""
#         pass


#     def step(self, gradients):
#         """Apply gradients to parameters."""
#         self.optimizer.apply_gradients(zip(gradients, self.params))







# Testing torch_optim_Adam vs torch.optim.Adam
def test_torch_and_tf_adam():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    tf.random.set_seed(42)
    
    np.random.seed(42)

    

    # Define a simple model in PyTorch
    torch_model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

    # Define the same model in TensorFlow
    tf_model = nn_Sequential([
        # tf.keras.layers.Dense(5, activation='relu', input_shape=(10,)),
        # tf.keras.layers.Dense(1)
        nn_Linear(10, 5),
        nn_ReLU(),
        nn_Linear(5, 1)
    ])

    # tf_model.build( input_shape = (None, 10) )
    _ = tf_model(tf.constant(np.random.randn(1, 10).astype(np.float32)))

    # Initialize weights in TensorFlow model to match PyTorch
    for torch_layer, tf_layer in zip(torch_model, tf_model):
        if isinstance(torch_layer, nn.Linear):
            # tf_layer.model.kernel.assign(tf.convert_to_tensor(torch_layer.weight.data.numpy().T, dtype=tf.float32))
            # tf_layer.model.bias.assign(tf.convert_to_tensor(torch_layer.bias.data.numpy(), dtype=tf.float32))
            print("tf_layer = ", tf_layer)

            tf_layer.trainable_weights[0].assign(torch_layer.weight.detach().numpy().T)  # kernel
            tf_layer.trainable_weights[1].assign(torch_layer.bias.detach().numpy())     # bias



    # Define inputs and targets
    inputs = np.random.rand(4, 10).astype(np.float32)
    targets = np.random.rand(4, 1).astype(np.float32)

    # Define optimizers
    torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)
    tf_optimizer = torch_optim_Adam(tf_model.trainable_variables, lr=0.01)

    # Define loss functions
    torch_loss_fn = nn.MSELoss()
    tf_loss_fn = tf.keras.losses.MeanSquaredError()

    # Training loop
    for step in range(5):
        # PyTorch
        torch_inputs = torch.tensor(inputs)
        torch_targets = torch.tensor(targets)
        print("torch_targets.device = ", torch_targets.device)

        torch_optimizer.zero_grad()
        torch_outputs = torch_model(torch_inputs)
        torch_loss = torch_loss_fn(torch_outputs, torch_targets)
        torch_loss.backward()

        torch_optimizer.step()

        tf_inputs = tf.convert_to_tensor(inputs)
        tf_targets = tf.convert_to_tensor(targets)

        tf_loss, tf_gradients = model_forward_backward_gradients(tf_inputs, tf_targets, tf_loss_fn, tf_model)
        # # TensorFlow
        # with tf.GradientTape() as tape:
        #     tf_outputs = tf_model(inputs)
        #     tf_loss = tf_loss_fn(targets, tf_outputs)
        # tf_gradients = tape.gradient(tf_loss, tf_model.trainable_variables)

        tf_optimizer.step(tf_gradients)

        # Print losses
        print(f"Step {step + 1}:")
        print(f"  PyTorch Loss: {torch_loss.item():.6f}")
        print(f"  TensorFlow Loss: {tf_loss.numpy():.6f}")

        # print("np.allclose(torch_loss.item(), tf_loss.numpy()) = ", np.allclose(torch_loss.item(), tf_loss.numpy()) )
        assert np.allclose(torch_loss.item(), tf_loss.numpy(), atol = 1e-6)

    # Compare final outputs
    torch_final_output = torch_model(torch.tensor(inputs)).detach().numpy()
    tf_final_output = tf_model(inputs).numpy()


    print("\nFinal Output Comparison:")
    print(f"  PyTorch: {torch_final_output}")
    print(f"  TensorFlow: {tf_final_output}")
    # print("np.allclose(torch_final_output, tf_final_output) = ", np.allclose(torch_final_output, tf_final_output) )
    assert np.allclose(torch_final_output, tf_final_output)





# Run the test
test_torch_and_tf_adam()


