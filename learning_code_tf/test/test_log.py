import numpy as np

from util.torch_to_tf import torch_log, torch_tensor

a = np.array([ [1,2,3], [4,5,6] ])

import torch


torch_a = torch.tensor(a)

print(torch.log(torch_a))




tf_a = torch_tensor(a)

print(torch_log(tf_a))

