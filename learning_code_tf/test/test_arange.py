

from util.torch_to_tf import torch_arange


tensor = torch_arange(0, 10)
print(tensor)
# 输出：tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 创建从 1 到 9 的张量，步长为 2
tensor = torch_arange(1, 10, step=2)
print(tensor)


import torch

tensor = torch.arange(0, 10)
print(tensor)
# 输出：tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 创建从 1 到 9 的张量，步长为 2
tensor = torch.arange(1, 10, step=2)
print(tensor)