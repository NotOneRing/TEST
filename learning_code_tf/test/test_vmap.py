import torch

# torch.dot                            # [D], [D] -> []
batched_dot = torch.vmap(torch.vmap(torch.dot))  # [N1, N0, D], [N1, N0, D] -> [N1, N0]
x, y = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
print("x = ", x)
print("y = ", y)
result = batched_dot(x, y) # tensor of size [2, 3]

print("result = ", result)



# torch.dot                            # [N], [N] -> []
batched_dot = torch.vmap(torch.dot, in_dims=1)  # [N, D], [N, D] -> [D]
x, y = torch.randn(2, 5), torch.randn(2, 5)
print("x = ", x)
print("y = ", y)
result2 = batched_dot(x, y)   # output is [5] instead of [2] if batched along the 0th dimension

print("result2 = ", result2)






