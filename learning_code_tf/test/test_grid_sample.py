import torch
import tensorflow as tf
import numpy as np


def bilinear_interpolation(x, y, points):
    """
    进行双线性插值
    :param x: 目标点的x坐标
    :param y: 目标点的y坐标
    :param points: 四个邻近点的坐标和像素值 [(x0, y0, V0), (x1, y0, V1), (x0, y1, V2), (x1, y1, V3)]
    :return: 插值结果
    """
    (x0, y0, V0), (x1, y0, V1), (x0, y1, V2), (x1, y1, V3) = points
    
    # 计算水平插值
    fxy0 = V0 * (x1 - x) / (x1 - x0) + V1 * (x - x0) / (x1 - x0)
    fxy1 = V2 * (x1 - x) / (x1 - x0) + V3 * (x - x0) / (x1 - x0)
    
    # 计算垂直插值
    f = fxy0 * (y1 - y) / (y1 - y0) + fxy1 * (y - y0) / (y1 - y0)
    
    return f

# # 示例
# points = [(0, 0, 10), (1, 0, 20), (0, 1, 30), (1, 1, 40)]
# x, y = 0.5, 0.5  # 目标点
# result = bilinear_interpolation(x, y, points)
# print(result)  # 输出插值结果


def tf_grid_sample(input_tensor, grid):
    batch_size, height, width, channels = input_tensor.shape
    grid_x, grid_y = grid[..., 0], grid[..., 1]

    # 通过grid选择输入张量的对应位置
    grid_x = tf.clip_by_value(grid_x, 0.0, width-1.0)
    grid_y = tf.clip_by_value(grid_y, 0.0, height-1.0)
    
    grid_x = tf.cast(grid_x, tf.int32)
    grid_y = tf.cast(grid_y, tf.int32)

    gathered_values = tf.gather(input_tensor, grid_x, axis=2, batch_dims=1)
    gathered_values = tf.gather(gathered_values, grid_y, axis=1, batch_dims=1)

    return gathered_values

# 固定种子以确保相同的随机数
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 创建相同的输入张量（[batch_size, height, width, channels]）
input_tensor_torch = torch.rand(1, 32, 32, 3)  # Batch size 1, 32x32 image, 3 channels

input_tensor_tf = tf.convert_to_tensor(input_tensor_torch.numpy())

input_tensor_torch = input_tensor_torch.permute(0, 3, 1, 2)  # Convert to NCHW format for PyTorch

# input_tensor_tf = tf.random.normal([1, 32, 32, 3])  # TensorFlow uses NHWC format

# 创建相同的网格张量（[batch_size, height, width, 2]）
grid_torch = torch.rand(1, 32, 32, 2)  # Batch size 1, height 32, width 32, 2 for (x, y) coordinates
# grid_tf = tf.random.normal([1, 32, 32, 2])  # TensorFlow uses (x, y) coordinates for grid

grid_tf = tf.convert_to_tensor(grid_torch.numpy())

# 使用grid_sample进行采样（PyTorch）
output_torch = torch.nn.functional.grid_sample(input_tensor_torch, grid_torch)

# 使用tf.image.grain_sample进行采样（TensorFlow）
output_tf = tf_grid_sample(input_tensor_tf, grid_tf)

# 比较输出
torch_output = output_torch.detach().numpy()
tensorflow_output = output_tf.numpy()


print("torch_output.shape = ", torch_output.shape)
print("tensorflow_output.shape = ", tensorflow_output.shape)


tensorflow_output = np.transpose(tensorflow_output, (0, 3, 1, 2))  # 转换为 (1, 3, 32, 32)







# 检查二者是否一致
if np.allclose(torch_output, tensorflow_output, atol=1e-5):
    print("Outputs are close enough.")
else:
    print("Outputs are different.")






























