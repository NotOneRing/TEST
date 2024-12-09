import torch
import torch.nn as nn
from torch.func import stack_module_state

# 定义简单模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

temp = [SimpleNet()]

stacked_params, stacked_buffers = stack_module_state(temp)

print("Stacked Parameters Shape:", {k: v.shape for k, v in stacked_params.items()})
print("Stacked Buffers:", stacked_buffers)


# 创建多个模型实例
models = [SimpleNet() for _ in range(3)]

# 堆叠这些模型的参数和缓冲区
stacked_params, stacked_buffers = stack_module_state(models)

print("Stacked Parameters Shape:", {k: v.shape for k, v in stacked_params.items()})
print("Stacked Buffers:", stacked_buffers)


# W的维度是(out_features, in_features)，因为y = xW^{\top} + b








































