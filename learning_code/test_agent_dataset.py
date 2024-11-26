import os
import pytest
import torch
from agent.dataset.sequence import StitchedSequenceDataset  # 确保正确导入


@pytest.fixture
def pretrain_dataset():
    # 解析 YAML 配置文件中的参数
    train_dataset_path = "/ssddata/qtguo/GENERAL_DATA/gym/hopper-medium-v2/train.npz"
    horizon_steps = 4
    cond_steps = 1
    # device = "cuda:0"
    device = "cuda"

    # 初始化数据集
    dataset = StitchedSequenceDataset(
        dataset_path=train_dataset_path,
        horizon_steps=horizon_steps,
        cond_steps=cond_steps,
        device=device,
    )
    return dataset


def test_pretrain_dataset_initialization(pretrain_dataset):
    # 检查数据集是否正确初始化
    assert pretrain_dataset.states is not None, "States should not be None."
    assert pretrain_dataset.actions is not None, "Actions should not be None."
    assert len(pretrain_dataset) > 0, "Dataset should not be empty."
    print(f"Dataset contains {len(pretrain_dataset)} samples.")


def test_pretrain_dataset_get_item(pretrain_dataset):
    # 测试获取单个样本
    batch = pretrain_dataset[0]  # 获取第一个样本

    assert isinstance(batch, dict), f"Expected batch to be a dict, but got {type(batch)}."
    assert "actions" in batch, "Batch should contain 'actions'."
    assert "conditions" in batch, "Batch should contain 'conditions'."
    assert isinstance(batch["actions"], torch.Tensor), "Actions should be a PyTorch tensor."
    assert isinstance(batch["conditions"], dict), "Conditions should be a dictionary."

    print(f"Batch actions shape: {batch['actions'].shape}")
    print(f"Batch conditions: {batch['conditions']}")


def test_pretrain_dataset_split(pretrain_dataset):
    # 测试数据集的训练-验证划分
    train_ratio = 0.8
    val_indices = pretrain_dataset.set_train_val_split(train_split=train_ratio)

    train_size = len(pretrain_dataset.indices)
    val_size = len(val_indices)

    total_size = train_size + val_size
    expected_train_size = int(total_size * train_ratio)

    assert train_size == expected_train_size, f"Expected train size {expected_train_size}, but got {train_size}."
    assert val_size == total_size - train_size, f"Validation size mismatch. Got {val_size}."


def test_pretrain_dataset_length(pretrain_dataset):
    # 测试数据集长度
    length = len(pretrain_dataset)
    print(f"Dataset length: {length}")
    assert length > 0, "Dataset length should be greater than 0."
