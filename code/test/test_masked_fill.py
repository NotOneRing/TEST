import torch
import numpy as np

def test_masked_fill_():
    # test cases list
    test_cases = [
        # 1D tensor and 1D mask
        {"tensor_shape": (5,), "mask_shape": (5,), "fill_value": -1},
        
        # 2D tensor and matched mask
        {"tensor_shape": (3, 4), "mask_shape": (3, 4), "fill_value": 99},
        
        # 2D tensor and broadcastable 1D mask
        {"tensor_shape": (3, 4), "mask_shape": (4,), "fill_value": -999},
        
        # 3D tensor and matched mask
        {"tensor_shape": (2, 3, 4), "mask_shape": (2, 3, 4), "fill_value": 7},
        
        # 3D tensor and broadcastable 2D mask
        {"tensor_shape": (2, 3, 4), "mask_shape": (3, 4), "fill_value": 0},
        
        # 3D tensor and broadcastable 1D mask
        {"tensor_shape": (2, 3, 4), "mask_shape": (4,), "fill_value": 42},
        
        # boundary test: 1D tensor, mask is True for all values
        {"tensor_shape": (5,), "mask_shape": (5,), "fill_value": 1e9, "custom_mask": torch.tensor([True, True, True, True, True])},
        
        # boundary test: 3D tensor，mask is False for all values
        {"tensor_shape": (2, 3, 4), "mask_shape": (2, 3, 4), "fill_value": -1e9, "custom_mask": torch.zeros((2, 3, 4), dtype=torch.bool)},
    ]
    
    for i, case in enumerate(test_cases):
        # get tensor and mask shape
        tensor_shape = case["tensor_shape"]
        mask_shape = case["mask_shape"]
        fill_value = case["fill_value"]
        custom_mask = case.get("custom_mask", None)

        # create random tensor
        tensor = torch.rand(tensor_shape)

        # create random mask (if no custom_mask is provided）
        if custom_mask is None:
            mask = torch.randint(0, 2, mask_shape, dtype=torch.bool)  # generate bool mask randomly
        else:
            mask = custom_mask

        # log original tensor for comparison
        original_tensor = tensor.clone()

        # run masked_fill_
        tensor.masked_fill_(mask, fill_value)

        # print test results
        print(f"Test case {i + 1}:")
        print(f"Tensor shape: {tensor_shape}, Mask shape: {mask_shape}")
        print(f"Mask (broadcastable): {np.array_equal(mask.shape, tensor.shape) or mask.shape == tensor.shape}")
        print(f"Original tensor:\n{original_tensor}")
        print(f"Mask:\n{mask}")
        print(f"Tensor after masked_fill_:\n{tensor}\n")

        # check if filled values are correct
        expected_tensor = original_tensor.clone()
        # expected_tensor[mask] = fill_value
        # broadcasted_mask = mask if mask.shape == tensor.shape else mask.unsqueeze(0).expand_as(tensor)
        broadcasted_mask = mask if mask.shape == tensor.shape else mask.expand_as(tensor)

        expected_tensor[broadcasted_mask] = fill_value

        assert torch.allclose(tensor, expected_tensor), f"Test case {i + 1} failed!"
        print(f"Test case {i + 1} passed!\n{'-' * 40}\n")

# run test
test_masked_fill_()



























