import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
action_opt_loss = x**2  # action_opt_loss is a tensor [1.0, 4.0, 9.0]

# calculate gradient
action_opt_loss.backward(torch.ones_like(action_opt_loss))

# check gradient
print(x.grad)  # output [2.0, 4.0, 6.0]



x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
action_opt_loss = x**2  # action_opt_loss is a tensor: [1.0, 4.0, 9.0]

# action_opt_loss = torch.sum(x**2)  # action_opt_loss is a tensor: [1.0, 4.0, 9.0]

# calculate gradient
action_opt_loss.backward()

# check gradient
print(x.grad)  # output [2.0, 4.0, 6.0]



