

import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

print(torch.cuda.current_device())  # current device


# check if there are currently available GPUs
print("Available GPUs:", torch.cuda.device_count())

# create a tensor in the GPU
tensor_gpu = torch.rand((3, 3), device='cuda:0')  # if there exists no GPU, it will choose CPU automatically
print(f"Tensor on device: {tensor_gpu.device}")  # print device information

# migrate tensor to CPU
tensor_cpu = tensor_gpu.cpu()

# print information of the migrated tensor and its device
print(f"Tensor on device: {tensor_cpu.device}\n{tensor_cpu}")





a = torch.tensor([0], dtype=torch.int64)

b = torch.linspace(0, 20, 20).reshape(2, 2, 5)

print("a = ", a)

print("b = ", b)

print("b[a] = ", b[a])

print("b[a, ...] = ", b[a, ...])


# c = torch.gather(b, 0, a)


# print("c = ", c)



c = torch.linspace(0, 20, 20).reshape(2, 2, 5)

print("c[0]", c[0])

print("c[0].shape", c[0].shape)



print("c[a]", c[a])

print("c[a].shape", c[a].shape)
















