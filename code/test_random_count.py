import numpy as np
from functools import wraps

# Initialize counter
random_call_count = 0

# Define decorator
def count_random_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global random_call_count
        random_call_count += 1
        return func(*args, **kwargs)
    return wrapper

# Hook np.random functions (example: Hook two functions)
np.random.normal = count_random_calls(np.random.normal)
np.random.randint = count_random_calls(np.random.randint)

# Test the calls
data1 = np.random.normal(0, 1, size=10)
data2 = np.random.randint(0, 10, size=5)

print(f"Random functions called {random_call_count} times.")

