import torch
import numpy as np
import unittest


class TestMaskedFill(unittest.TestCase):
    def setUp(self):
        # Define test cases that will be used by all test methods
        self.test_cases = [
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
            {"tensor_shape": (5,), "mask_shape": (5,), "fill_value": 1e9, 
             "custom_mask": torch.tensor([True, True, True, True, True])},
            
            # boundary test: 3D tensor，mask is False for all values
            {"tensor_shape": (2, 3, 4), "mask_shape": (2, 3, 4), "fill_value": -1e9, 
             "custom_mask": torch.zeros((2, 3, 4), dtype=torch.bool)},
        ]

    def test_masked_fill(self):
        """Test PyTorch's masked_fill_ operation with various tensor and mask shapes."""
        for i, case in enumerate(self.test_cases):
            with self.subTest(i=i, case=case):
                # Get tensor and mask shape
                tensor_shape = case["tensor_shape"]
                mask_shape = case["mask_shape"]
                fill_value = case["fill_value"]
                custom_mask = case.get("custom_mask", None)

                # Create random tensor
                tensor = torch.rand(tensor_shape)

                # Create random mask (if no custom_mask is provided)
                if custom_mask is None:
                    mask = torch.randint(0, 2, mask_shape, dtype=torch.bool)  # Generate bool mask randomly
                else:
                    mask = custom_mask

                # Log original tensor for comparison
                original_tensor = tensor.clone()

                # Run masked_fill_
                tensor.masked_fill_(mask, fill_value)

                # Check if filled values are correct
                expected_tensor = original_tensor.clone()
                broadcasted_mask = mask if mask.shape == tensor.shape else mask.expand_as(tensor)
                expected_tensor[broadcasted_mask] = fill_value

                self.assertTrue(
                    torch.allclose(tensor, expected_tensor),
                    f"Test case {i + 1} failed! Tensor shape: {tensor_shape}, Mask shape: {mask_shape}"
                )

    def test_all_true_mask(self):
        """Test masked_fill_ with a mask that is True for all values."""
        tensor = torch.rand(5)
        mask = torch.tensor([True, True, True, True, True])
        fill_value = 1e9
        
        original_tensor = tensor.clone()
        tensor.masked_fill_(mask, fill_value)
        
        # All values should be replaced with fill_value
        expected_tensor = torch.full_like(original_tensor, fill_value)
        self.assertTrue(torch.allclose(tensor, expected_tensor))

    def test_all_false_mask(self):
        """Test masked_fill_ with a mask that is False for all values."""
        tensor = torch.rand((2, 3, 4))
        mask = torch.zeros((2, 3, 4), dtype=torch.bool)
        fill_value = -1e9
        
        original_tensor = tensor.clone()
        tensor.masked_fill_(mask, fill_value)
        
        # No values should be replaced
        self.assertTrue(torch.allclose(tensor, original_tensor))

    def test_broadcasting_1d_mask_to_2d_tensor(self):
        """Test masked_fill_ with broadcasting a 1D mask to a 2D tensor."""
        tensor = torch.rand((3, 4))
        mask = torch.tensor([False, True, False, True])  # Will be broadcast to each row
        fill_value = -999.0
        
        original_tensor = tensor.clone()
        tensor.masked_fill_(mask, fill_value)

        # print("original_tensor = ", original_tensor)
        # print("tensor = ", tensor)
        
        # Check specific columns (1 and 3) are filled
        for i in range(3):  # For each row
            self.assertTrue(torch.allclose(tensor[i, 0], original_tensor[i, 0]))
            self.assertTrue(torch.allclose(tensor[i, 1], torch.tensor(fill_value)))
            self.assertTrue(torch.allclose(tensor[i, 2], original_tensor[i, 2]))
            self.assertTrue(torch.allclose(tensor[i, 3], torch.tensor(fill_value)))


if __name__ == '__main__':
    unittest.main()
