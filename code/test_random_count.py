import numpy as np
from functools import wraps

# 计数器初始化
random_call_count = 0

# 定义装饰器
def count_random_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global random_call_count
        random_call_count += 1
        return func(*args, **kwargs)
    return wrapper

# Hook 所有 np.random 函数（示例 Hook 两个函数）
np.random.normal = count_random_calls(np.random.normal)
np.random.randint = count_random_calls(np.random.randint)

# 测试调用
data1 = np.random.normal(0, 1, size=10)
data2 = np.random.randint(0, 10, size=5)

print(f"Random functions called {random_call_count} times.")