

import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

print(torch.cuda.current_device())  # 当前设备


# 检查当前是否有可用的 GPU
print("Available GPUs:", torch.cuda.device_count())

# 在 GPU 上创建一个张量
tensor_gpu = torch.rand((3, 3), device='cuda:0')  # 如果没有 GPU，会默认使用 CPU
print(f"Tensor on device: {tensor_gpu.device}")  # 显示设备信息

# 将张量迁移到 CPU
tensor_cpu = tensor_gpu.cpu()

# 打印迁移后的张量及其设备信息
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
















