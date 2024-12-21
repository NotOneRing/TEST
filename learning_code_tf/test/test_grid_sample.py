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


def tf_grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    assert padding_mode == "zeros", "only zeros padding_mode is implemented right now"
    assert mode == "bilinear", "only bilinear is implemented right now"
    assert len(input.shape) == 4, "len(input.shape) must be 4"
    batch_size, channels, height_in, width_in = input.shape

    batch_grid, height_out, width_out, _ = grid.shape

    print("batch_size = ", batch_size)

    print("channels = ", channels)

    print("height_in = ", height_in)

    print("width_in = ", width_in)

    print("batch_grid = ", batch_grid)

    print("height_out = ", height_out)

    print("width_out = ", width_out)

    assert batch_size == batch_grid, "input has the same batch size as grid"

    grid_x, grid_y = grid[..., 0], grid[..., 1]


    grid_x_rescaled = (grid_x + 1) / 2 * (width_in - 1)
    # grid_x_ceil_pos = (grid_x_ceil + 1) / 2 * (width_in - 1)
    
    grid_y_rescaled = (-grid_y + 1) / 2 * (height_in - 1)    
    # grid_y_ceil_pos = (-grid_y_ceil + 1) / 2 * (height_in - 1)


    grid_x_ceil = tf.math.ceil(grid_x_rescaled)
    grid_x_floor = tf.math.floor(grid_x_rescaled)

    grid_y_ceil = tf.math.ceil(grid_y_rescaled)
    grid_y_floor = tf.math.floor(grid_y_rescaled)



    result_tensor = np.zeros((batch_size, channels, height_out, width_out))

    for batch_index in range(batch_size):
        for h_out in range(height_out):
            for w_out in range(width_out):
                x_floor = grid_x_floor[batch_index, h_out, w_out]
                x_ceil = grid_x_ceil[batch_index, h_out, w_out]
                y_floor = grid_y_floor[batch_index, h_out, w_out]
                y_ceil = grid_y_ceil[batch_index, h_out, w_out]
                x_floor = tf.cast(x_floor, tf.int32).numpy().item()
                x_ceil = tf.cast(x_ceil, tf.int32).numpy().item()
                y_floor = tf.cast(y_floor, tf.int32).numpy().item()
                y_ceil = tf.cast(y_ceil, tf.int32).numpy().item()


                x_index = (grid_x[batch_index, h_out, w_out].numpy().item() + 1) / 2 * (width_in - 1)
                y_index = (grid_y[batch_index, h_out, w_out].numpy().item() + 1) / 2 * (height_in - 1)    

                if h_out == 1 and w_out == 1:
                    print("x_index = ", x_index)
                    print("y_index = ", y_index)

                    print("x_floor = ", x_floor)
                    print("x_ceil = ", x_ceil)
                    print("y_floor = ", y_floor)
                    print("y_ceil = ", y_ceil)
                    print(" = ", input[batch_index, :, y_floor, x_floor])
                    print(" = ", input[batch_index, :, y_floor, x_ceil])
                    print(" = ", input[batch_index, :, y_ceil, x_floor])
                    print(" = ", input[batch_index, :, y_ceil, x_ceil])

                point0 = (x_floor, y_floor, input[batch_index, :, y_floor, x_floor])


                point1 = (x_ceil, y_floor, input[batch_index, :, y_floor, x_ceil])
                point2 = (x_floor, y_ceil, input[batch_index, :, y_ceil, x_floor])
                point3 = (x_ceil, y_ceil, input[batch_index, :, y_ceil, x_ceil])

                # print("point0[2].shape = ", point0[2].shape)


                bilinear_result = bilinear_interpolation(
                    x_index, y_index,
                    [
                        point0, point1, point2, point3
                    ]
                )

                result_tensor[batch_index, :, h_out, w_out] = bilinear_result.numpy()
                
    result_tensor = tf.convert_to_tensor(result_tensor)

    return result_tensor
            
    # grid_x_floor_pos.shape = batch_size, height_out, width_out
    #接下来，可以给进去batch_size的index，height_in和width_in的index，
    #然后根据这个index得到[batch_size, channels, height_out, width_out]的output

    # output.shape = batch_size, channels, height_out, width_out


# 固定种子以确保相同的随机数
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 创建相同的输入张量（[batch_size, height, width, channels]）
# input_tensor_torch = torch.rand(1, 32, 32, 3)  # Batch size 1, 32x32 image, 3 channels

# input_tensor_torch = torch.range(start = 1, end = 1*32*32*3).reshape(1, 32, 32, 3)

input_tensor_torch = torch.range(start = 1, end = 1*3*3*3).reshape(1, 3, 3, 3)

input_tensor_torch = input_tensor_torch.permute(0, 3, 1, 2)  # Convert to NCHW format for PyTorch

input_tensor_tf = tf.convert_to_tensor(input_tensor_torch.numpy())

# input_tensor_tf = tf.random.normal([1, 32, 32, 3])  # TensorFlow uses NHWC format

# 创建相同的网格张量（[batch_size, height, width, 2]）
# grid_torch = torch.rand(1, 32, 32, 2)  # Batch size 1, height 32, width 32, 2 for (x, y) coordinates

# grid_tf = tf.random.normal([1, 32, 32, 2])  # TensorFlow uses (x, y) coordinates for grid

# grid_torch = torch.range(start = 1, end = 1*32*32*2).reshape(1, 32, 32, 2)
grid_torch = torch.range(start = 1, end = 1*3*3*2).reshape(1, 3, 3, 2)

# grid_torch = (grid_torch - 1*32*32*2  / 2) / (1*32*32*2)
grid_torch = (grid_torch - 1*3*3*2  / 2) / (1*3*3*2)

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





print("torch_output = ", torch_output)
print("tensorflow_output = ", tensorflow_output)





# 使用 np.isnan() 检查 NaN 的位置
nan_positions = np.where(np.isnan(tensorflow_output))

print("NaN 位置的索引:", nan_positions)

print("tensorflow_output[0, 0, 1, 1] = ", tensorflow_output[0, 0, 1, 1])
print("tensorflow_output[0, 1, 1, 1] = ", tensorflow_output[0, 1, 1, 1])
print("tensorflow_output[0, 2, 1, 1] = ", tensorflow_output[0, 2, 1, 1])








# 检查二者是否一致
if np.allclose(torch_output, tensorflow_output, atol=1e-5):
    print("Outputs are close enough.")
else:
    print("Outputs are different.")






























