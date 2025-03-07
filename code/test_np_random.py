
# import numpy as np

# import random

# random.seed(42)
# np.random.seed(42)

# # a = np.random.randn(1, 10)


# # print("a = ", a)
# n = 10
# c = np.random.permutation(n)

# print("c = ", c)


import numpy as np

# Set the seed for the first time and generate random numbers
np.random.seed(0)
print("First random numbers:", np.random.rand(3))
# Output: [0.5488135  0.71518937 0.60276338]

# Set the same seed again and generate random numbers
np.random.seed(0)
print("Second random numbers:", np.random.rand(3))
# Output: [0.5488135  0.71518937 0.60276338]







