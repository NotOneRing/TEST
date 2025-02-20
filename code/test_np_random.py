
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

# 第一次设置种子并生成随机数
np.random.seed(0)
print("第一次随机数:", np.random.rand(3))
# 输出: [0.5488135  0.71518937 0.60276338]

# 第二次设置相同种子并生成随机数
np.random.seed(0)
print("第二次随机数:", np.random.rand(3))
# 输出: [0.5488135  0.71518937 0.60276338]