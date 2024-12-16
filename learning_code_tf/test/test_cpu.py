# import tensorflow as tf

# # 检查当前是否有可用的 GPU
# print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# # 在 GPU 上创建一个张量
# with tf.device('/GPU:0'):  # 如果没有 GPU，会默认使用 CPU
#     tensor_gpu = tf.random.uniform(shape=(3, 3), dtype=tf.float32)

# # 打印张量的设备信息
# print(f"Tensor on device: {tensor_gpu.device}")  # 显示设备信息

# # 将张量迁移到 CPU
# tensor_cpu = tensor_gpu.numpy()  # .numpy() 会自动迁移到 CPU

# print("type(tensor_cpu) = ", type(tensor_cpu))

# # 打印迁移后的张量及其设备信息
# print(f"Tensor on device: CPU\n{tensor_cpu}")





















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


























