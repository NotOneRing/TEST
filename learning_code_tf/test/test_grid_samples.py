import tensorflow as tf
import torch
import torch.nn.functional as F
import numpy as np

def bilinear_interpolate(image, grid):
    """
    双线性插值实现。
    :param image: 输入图像 (N, H, W, C)
    :param grid: 采样网格 (N, H_out, W_out, 2)，像素坐标
    :return: 输出图像 (N, H_out, W_out, C)
    """
    n, h, w, c = image.shape
    _, h_out, w_out, _ = grid.shape

    # 分解采样网格
    x, y = grid[..., 0], grid[..., 1]

    # 获取最近的整数坐标
    x0, x1 = tf.floor(x), tf.floor(x + 1)
    y0, y1 = tf.floor(y), tf.floor(y + 1)

    # 限制坐标范围
    x0 = tf.clip_by_value(tf.cast(x0, tf.int32), 0, w - 1)
    x1 = tf.clip_by_value(tf.cast(x1, tf.int32), 0, w - 1)
    y0 = tf.clip_by_value(tf.cast(y0, tf.int32), 0, h - 1)
    y1 = tf.clip_by_value(tf.cast(y1, tf.int32), 0, h - 1)

    # 添加批次索引
    batch_idx = tf.range(n)[:, None, None]
    batch_idx = tf.tile(batch_idx, [1, h_out, w_out])

    # 获取四个角的像素值
    Ia = tf.gather_nd(image, tf.stack([batch_idx, y0, x0], axis=-1))
    Ib = tf.gather_nd(image, tf.stack([batch_idx, y1, x0], axis=-1))
    Ic = tf.gather_nd(image, tf.stack([batch_idx, y0, x1], axis=-1))
    Id = tf.gather_nd(image, tf.stack([batch_idx, y1, x1], axis=-1))

    # 计算插值权重
    x0, x1 = tf.cast(x0, tf.float32), tf.cast(x1, tf.float32)
    y0, y1 = tf.cast(y0, tf.float32), tf.cast(y1, tf.float32)
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # 合并加权值
    return tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    TensorFlow 的 grid_sample 实现，匹配 PyTorch 行为。
    """
    if align_corners:
        # PyTorch 的 align_corners=True 时需要线性映射到 [-0.5, size - 0.5]
        grid = (grid + 1) / 2 * (tf.constant(input.shape[1:3], dtype=tf.float32) - 1)

    # 检查 padding_mode
    if padding_mode == 'border':
        grid = tf.clip_by_value(grid, 0.0, tf.constant(input.shape[1:3], dtype=tf.float32) - 1)
    elif padding_mode == 'zeros':
        # TensorFlow 默认处理
        pass

    # 使用双线性插值
    output = bilinear_interpolate(input, grid)

    # 对超出边界的值手动设置为 0 (zeros padding)
    if padding_mode == 'zeros':
        mask = tf.reduce_all((grid >= 0) & (grid < tf.constant(input.shape[1:3], dtype=tf.float32)), axis=-1)
        mask = tf.cast(mask, dtype=input.dtype)
        output = output * tf.expand_dims(mask, axis=-1)

    return output

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F

# 创建输入图像和采样网格
image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=np.float32).reshape(1, 3, 3, 1)  # (N, H, W, C)

grid = np.array([
    [
        [[-1, -1], [1, -1]],  # 左上角到右上角
        [[-1, 1], [1, 1]]  # 左下角到右下角
    ]
], dtype=np.float32)  # 归一化坐标 [-1, 1]

# TensorFlow 测试
image_tf = tf.convert_to_tensor(image)
grid_tf = tf.convert_to_tensor(grid)
output_tf = grid_sample(image_tf, grid_tf, align_corners=True)

# PyTorch 测试
image_torch = torch.tensor(image.transpose(0, 3, 1, 2))  # (N, C, H, W)
grid_torch = torch.tensor(grid)
output_torch = F.grid_sample(image_torch, grid_torch, align_corners=True)

# 对比结果
output_tf_np = output_tf.numpy().transpose(0, 3, 1, 2)  # 转换为 (N, C, H, W)
print("TensorFlow Output:\n", output_tf_np)
print("PyTorch Output:\n", output_torch.numpy())
assert np.allclose(output_tf_np, output_torch.numpy(), atol=1e-5), "Outputs do not match!"
