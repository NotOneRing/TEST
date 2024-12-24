import torch
import torch.optim as optim
import torch.nn as nn

# from util.scheduler import CosineAnnealingWarmupRestarts

import math

from util.torch_to_tf import tf_CosineAnnealingWarmupRestarts


# From https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: -1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
    last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):

        print("scheduler.py: CosineAnnealingWarmupRestarts.__init__()")

        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):

        print("scheduler.py: CosineAnnealingWarmupRestarts.init_lr()")

        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):

        print("scheduler.py: CosineAnnealingWarmupRestarts.get_lr()")

        print("self.base_lrs = ", self.base_lrs)



        if self.step_in_cycle == -1:
            print("get_lr: branch1")
            result =  self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            print("get_lr: branch2")
            result = [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            print("get_lr: branch3")
            result = [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

        print("result = ", result)
        
        return result
        
    def step(self, epoch=None):

        print("scheduler.py: CosineAnnealingWarmupRestarts.step()")
        print("step: epoch = ", epoch)
        print("torch: self.last_epoch = ", self.last_epoch)

        if epoch is None:
            print("step: branch1")
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                print("step: branch1-1")
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            print("step: branch2")
            if epoch >= self.first_cycle_steps:
                print("step: branch2-1")
                if self.cycle_mult == 1.0:
                    print("step: branch2-1-1")
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    print("step: branch2-1-2")
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n

                    print("self.cycle = ", self.cycle)

                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )

                    print("self.step_in_cycle = ", self.step_in_cycle)

                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )

                    print("self.cur_cycle_steps = ", self.cur_cycle_steps)

            else:
                print("step: branch2-2")
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr




def test_learning_rate():
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

    # lr_epoch = lr_schedule.step()
    # lr_epoch = lr_schedule.step()

    # Define a dummy optimizer (just for example)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
 
    # Initialize the model (dummy model for testing)
    model = SimpleModel()

    from util.torch_to_tf import torch_optim_Adam
    optimizer = torch_optim_Adam(model.trainable_variables, lr=lr_schedule)


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
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


        # lr_epoch = lr_schedule(epoch)
        lr_epoch = lr_schedule.step()
        # lr_epoch = optimizer.learning_rate.numpy()

        # lr_epoch = optimizer.get_learning_rate()

        # .numpy()

        # Print learning rate using the lr_schedule directly
        print(f"Epoch {epoch+1}, Learning rate: {lr_epoch}")  # Accessing the learning rate value
        tf_lr_list.append(lr_epoch)


    for i in range(10):
        print("pytorch_lr_list[{}] = {}".format(i, pytorch_lr_list[i]))
        print("tf_lr_list[{}] = {}".format(i, tf_lr_list[i]))


    pytorch_result_arr_np = np.array(pytorch_lr_list)
    tf_result_arr_np = np.array(tf_lr_list)


    assert np.allclose(pytorch_result_arr_np, tf_result_arr_np)






















import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np

from util.torch_to_tf import nn_Sequential, nn_Linear, nn_ReLU, torch_optim_Adam, model_forward_backward_gradients






# Testing torch_optim_Adam vs torch.optim.Adam
def test_torch_and_tf_learning_rate_model():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    tf.random.set_seed(42)

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
    # torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=1)
    # torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.0000001)
    #这里的lr是废的，改多大多小都行，最后被CosineAnnealingWarmupRestarts里面的值替代了
    torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)

    # from util.scheduler import CosineAnnealingWarmupRestarts
    # Initialize the scheduler
    scheduler = CosineAnnealingWarmupRestarts(torch_optimizer, first_cycle_steps=5, cycle_mult=2.0, max_lr=0.1, min_lr=0.001, warmup_steps=2)


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

    # lr_epoch = lr_schedule.step()

    print("")
    print("before pass to optimizer")
    print("")

    tf_optimizer = torch_optim_Adam(tf_model.trainable_variables, lr=lr_schedule)

    print("")
    print("after pass to optimizer")
    print("")

    # Define loss functions
    torch_loss_fn = nn.MSELoss()
    tf_loss_fn = tf.keras.losses.MeanSquaredError()

    print("")
    print("Enter Loop")
    print("")

    # Training loop
    for out_step in range(3):
        for step in range(5):
            # PyTorch
            torch_inputs = torch.tensor(inputs)
            torch_targets = torch.tensor(targets)

            torch_optimizer.zero_grad()
            torch_outputs = torch_model(torch_inputs)
            torch_loss = torch_loss_fn(torch_outputs, torch_targets)
            torch_loss.backward()

            torch_optimizer.step()


            tf_loss, tf_gradients = model_forward_backward_gradients(inputs, targets, tf_loss_fn, tf_model)
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
            assert np.allclose(torch_loss.item(), tf_loss.numpy(), atol = 1e-5)

        scheduler.step()
        lr_epoch = lr_schedule.step()

    # Compare final outputs
    torch_final_output = torch_model(torch.tensor(inputs)).detach().numpy()
    tf_final_output = tf_model(inputs).numpy()

    print("\nFinal Output Comparison:")
    print(f"  PyTorch: {torch_final_output}")
    print(f"  TensorFlow: {tf_final_output}")
    # print("np.allclose(torch_final_output, tf_final_output) = ", np.allclose(torch_final_output, tf_final_output) )
    assert np.allclose(torch_final_output, tf_final_output, atol = 1e-5)



test_learning_rate()




print("Test Cases 2:")
print("")
print("")

# Run the test
test_torch_and_tf_learning_rate_model()





















