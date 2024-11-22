import pytest
import torch
import numpy as np

from sampling import extract

# Test case 1: Basic functionality
def test_extract_basic():
    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t = torch.tensor([0, 1, 2])
    x_shape = (3, 3)
    expected_output = torch.tensor([[1], [5], [9]])
    extracted = extract(a, t, x_shape)
    print("extracted = ", extracted)
    assert torch.all(extracted == expected_output)

# Test case 2: Different t values
def test_extract_different_t():
    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t = torch.tensor([1, 0, 2])
    x_shape = (3, 3)
    expected_output = torch.tensor([[2], [1], [9]])
    assert torch.all(extract(a, t, x_shape) == expected_output)

# Test case 3: Different a and x_shape
def test_extract_different_a_x_shape():
    a = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    t = torch.tensor([0, 1, 2])
    x_shape = (3, 2)
    expected_output = torch.tensor([[10], [50], [90]])
    assert torch.all(extract(a, t, x_shape) == expected_output)

# Test case 4: Empty a
def test_extract_empty_a():
    a = torch.tensor([])
    t = torch.tensor([0, 1, 2])
    x_shape = (3, 3)
    with pytest.raises(IndexError):
        extract(a, t, x_shape)

# Test case 5: Invalid t values
def test_extract_invalid_t():
    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t = torch.tensor([-1, 3, 4])
    x_shape = (3, 3)
    with pytest.raises(IndexError):
        extract(a, t, x_shape)


test_extract_basic()

# test_extract_different_a_x_shape()

# test_extract_different_t()

# test_extract_empty_a()

# test_extract_invalid_t()

