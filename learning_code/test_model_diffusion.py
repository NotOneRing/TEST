import pytest
import torch
import numpy as np



# sampling.py FINISHED
# model/diffusion/sampling.py
from model.diffusion.sampling import cosine_beta_schedule

def test_cosine_beta_schedule():
    # 测试参数
    timesteps = 10
    s = 0.008
    dtype = torch.float32

    # 调用函数
    betas = cosine_beta_schedule(timesteps, s, dtype)

    # 检查返回类型
    assert isinstance(betas, torch.Tensor), "Output is not a torch.Tensor"

    # 检查返回值的dtype
    assert betas.dtype == dtype, f"Expected dtype {dtype}, but got {betas.dtype}"

    # 检查返回值的长度是否正确
    assert len(betas) == timesteps, f"Expected length {timesteps}, but got {len(betas)}"

    # 检查beta值是否在 [0, 0.999] 范围内
    assert torch.all((betas >= 0) & (betas <= 0.999)), "Betas are out of expected range [0, 0.999]"

    # 检查beta值是否递增或合理分布
    alphas_cumprod = np.cos(((np.linspace(0, timesteps + 1, timesteps + 1) / (timesteps + 1)) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    expected_betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    expected_betas_clipped = np.clip(expected_betas, a_min=0, a_max=0.999)

    assert torch.allclose(betas, torch.tensor(expected_betas_clipped, dtype=dtype), atol=1e-6), \
        "Betas do not match the expected values"






from model.diffusion.sampling import extract

def test_extract():
    # Create test inputs
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)  # Shape: (2, 3)
    t = torch.tensor([[0], [2]], dtype=torch.int64)  # Shape: (2, 1)
    x_shape = (2, 1, 1)

    # Call the function
    output = extract(a, t, x_shape)

    # Expected output
    expected_output = torch.tensor([[[1]], [[6]]], dtype=torch.float32)  # Shape: (2, 1, 1)

    # Assertions
    assert output.shape == (2, 1, 1), "Output shape is incorrect."
    assert torch.allclose(output, expected_output), "Output values are incorrect."


from model.diffusion.sampling import make_timesteps

def test_make_timesteps():
    batch_size = 4
    i = 10
    device = torch.device("cuda")  # 确保测试在 GPU 上运行

    t = make_timesteps(batch_size, i, device)

    assert isinstance(t, torch.Tensor), "Output is not a torch.Tensor"
    assert t.dtype == torch.long, f"Expected dtype torch.long, but got {t.dtype}"
    assert t.shape == (batch_size,), f"Expected shape {(batch_size,)}, but got {t.shape}"
    assert torch.all(t == i), f"All elements in the tensor should be {i}, but got {t.tolist()}"
    assert t.device.type == device.type, f"Expected device type {device.type}, but got {t.device.type}"
    assert (t.device.index == device.index or device.index is None), \
        f"Expected device index {device.index}, but got {t.device.index}"


@pytest.mark.parametrize("batch_size, i, device", [
    (1, 0, torch.device("cpu")),
    (5, 42, torch.device("cpu")),
    (10, 7, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
])
def test_make_timesteps_parametrized(batch_size, i, device):
    t = make_timesteps(batch_size, i, device)
    assert isinstance(t, torch.Tensor), "Output is not a torch.Tensor"
    assert t.dtype == torch.long, f"Expected dtype torch.long, but got {t.dtype}"
    assert t.shape == (batch_size,), f"Expected shape {(batch_size,)}, but got {t.shape}"
    assert torch.all(t == i), f"All elements in the tensor should be {i}, but got {t.tolist()}"
    assert t.device.type == device.type, f"Expected device type {device.type}, but got {t.device.type}"
    assert (t.device.index == device.index or device.index is None), \
        f"Expected device index {device.index}, but got {t.device.index}"



