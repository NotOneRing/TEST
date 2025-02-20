from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

import torch

# 创建一个二维的 Normal 分布
loc = torch.zeros(3)  # 均值为 [0, 0, 0]
scale = torch.ones(3)  # 标准差为 [1, 1, 1]
normal = Normal(loc, scale)

print("loc.shape = ", loc.shape)

# 正常的 Normal 分布的 batch_shape 和 event_shape
print(normal.batch_shape)  # 输出: torch.Size([3])
print(normal.event_shape)  # 输出: torch.Size([])

# 使用 Independent 类，将 batch_shape 重新解释为 event_shape
independent_normal = Independent(normal, 1)

# 重新解释后的 batch_shape 和 event_shape
print(independent_normal.batch_shape)  # 输出: torch.Size([])
print(independent_normal.event_shape)  # 输出: torch.Size([3])



from torch.distributions import Normal, Independent

loc = torch.zeros(3)
scale = torch.ones(3)
normal = Normal(loc, scale)

# 重新解释第一个批次维度（3）为事件维度
independent_normal = Independent(normal, 1)

print(independent_normal.batch_shape)  # 输出 torch.Size([4])
print(independent_normal.event_shape)  # 输出 torch.Size([3, 2])
