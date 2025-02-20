import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
action_opt_loss = x**2  # action_opt_loss 是一个张量 [1.0, 4.0, 9.0]

# 计算梯度
action_opt_loss.backward(torch.ones_like(action_opt_loss))

# 查看梯度
print(x.grad)  # 输出 [2.0, 4.0, 6.0]



x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
action_opt_loss = x**2  # action_opt_loss 是一个张量 [1.0, 4.0, 9.0]

# action_opt_loss = torch.sum(x**2)  # action_opt_loss 是一个张量 [1.0, 4.0, 9.0]

# 计算梯度
action_opt_loss.backward()

# 查看梯度
print(x.grad)  # 输出 [2.0, 4.0, 6.0]



