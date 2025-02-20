import torch
import numpy as np

def test_masked_fill_():
    # 测试用例列表
    test_cases = [
        # 1D 张量与 1D mask
        {"tensor_shape": (5,), "mask_shape": (5,), "fill_value": -1},
        
        # 2D 张量与完全匹配的 mask
        {"tensor_shape": (3, 4), "mask_shape": (3, 4), "fill_value": 99},
        
        # 2D 张量与可广播的 1D mask
        {"tensor_shape": (3, 4), "mask_shape": (4,), "fill_value": -999},
        
        # 3D 张量与完全匹配的 mask
        {"tensor_shape": (2, 3, 4), "mask_shape": (2, 3, 4), "fill_value": 7},
        
        # 3D 张量与可广播的 2D mask
        {"tensor_shape": (2, 3, 4), "mask_shape": (3, 4), "fill_value": 0},
        
        # 3D 张量与可广播的 1D mask
        {"tensor_shape": (2, 3, 4), "mask_shape": (4,), "fill_value": 42},
        
        # 边界测试：1D 张量，mask 全为 True
        {"tensor_shape": (5,), "mask_shape": (5,), "fill_value": 1e9, "custom_mask": torch.tensor([True, True, True, True, True])},
        
        # 边界测试：3D 张量，mask 全为 False
        {"tensor_shape": (2, 3, 4), "mask_shape": (2, 3, 4), "fill_value": -1e9, "custom_mask": torch.zeros((2, 3, 4), dtype=torch.bool)},
    ]
    
    for i, case in enumerate(test_cases):
        # 获取张量和 mask 的形状
        tensor_shape = case["tensor_shape"]
        mask_shape = case["mask_shape"]
        fill_value = case["fill_value"]
        custom_mask = case.get("custom_mask", None)

        # 创建随机张量
        tensor = torch.rand(tensor_shape)

        # 创建随机 mask（如果没有提供自定义 mask）
        if custom_mask is None:
            mask = torch.randint(0, 2, mask_shape, dtype=torch.bool)  # 随机生成布尔 mask
        else:
            mask = custom_mask

        # 记录原始张量，用于比较
        original_tensor = tensor.clone()

        # 执行 masked_fill_
        tensor.masked_fill_(mask, fill_value)

        # 打印测试结果
        print(f"Test case {i + 1}:")
        print(f"Tensor shape: {tensor_shape}, Mask shape: {mask_shape}")
        print(f"Mask (broadcastable): {np.array_equal(mask.shape, tensor.shape) or mask.shape == tensor.shape}")
        print(f"Original tensor:\n{original_tensor}")
        print(f"Mask:\n{mask}")
        print(f"Tensor after masked_fill_:\n{tensor}\n")

        # 检查填充值是否正确
        expected_tensor = original_tensor.clone()
        # expected_tensor[mask] = fill_value
        # broadcasted_mask = mask if mask.shape == tensor.shape else mask.unsqueeze(0).expand_as(tensor)
        broadcasted_mask = mask if mask.shape == tensor.shape else mask.expand_as(tensor)

        expected_tensor[broadcasted_mask] = fill_value

        assert torch.allclose(tensor, expected_tensor), f"Test case {i + 1} failed!"
        print(f"Test case {i + 1} passed!\n{'-' * 40}\n")

# 执行测试
test_masked_fill_()



























