from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

import torch

# create a two-dimensional Normal distribution
loc = torch.zeros(3)  # the mean is [0, 0, 0]
scale = torch.ones(3)  # the std is [1, 1, 1]
normal = Normal(loc, scale)

print("loc.shape = ", loc.shape)

# Normal distribution's batch_shape and event_shape
print(normal.batch_shape)  # output: torch.Size([3])
print(normal.event_shape)  # output: torch.Size([])

# use Independent class, re-interpret batch_shape as event_shape
independent_normal = Independent(normal, 1)

# re-interpreted batch_shape and event_shape
print(independent_normal.batch_shape)  # output: torch.Size([])
print(independent_normal.event_shape)  # output: torch.Size([3])



from torch.distributions import Normal, Independent

loc = torch.zeros(3)
scale = torch.ones(3)
normal = Normal(loc, scale)

# re-interpret the first batch dimension (3) as the event dimension
independent_normal = Independent(normal, 1)

print(independent_normal.batch_shape)  # output torch.Size([4])
print(independent_normal.event_shape)  # output torch.Size([3, 2])
