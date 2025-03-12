# import tensorflow as tf

# # check if there exists available GPUs
# print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# # create a tensor in the GPU
# with tf.device('/GPU:0'):  # If no GPU exists, it use the CPU as the default device
#     tensor_gpu = tf.random.uniform(shape=(3, 3), dtype=tf.float32)

# # print tensor's device information
# print(f"Tensor on device: {tensor_gpu.device}")  # print device information

# # migrate tensor to CPU
# tensor_cpu = tensor_gpu.numpy()  # .numpy() will transfer to CPU automatically

# print("type(tensor_cpu) = ", type(tensor_cpu))

# # print migrated tensor and its device information
# print(f"Tensor on device: CPU\n{tensor_cpu}")





















import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

print(torch.cuda.current_device())  # current device


# check if there are currently available GPUs
print("Available GPUs:", torch.cuda.device_count())

# create a tensor in the GPU
tensor_gpu = torch.rand((3, 3), device='cuda:0')  # if there are no GPUs, it will choose the CPU automatically
print(f"Tensor on device: {tensor_gpu.device}")  # show device information

# migrate tensor to the CPU
tensor_cpu = tensor_gpu.cpu()

# print migrated tensor and its device information
print(f"Tensor on device: {tensor_cpu.device}\n{tensor_cpu}")


























