import pytest
import os
from omegaconf import OmegaConf
from agent.pretrain.train_agent import PreTrainAgent



# from omegaconf import OmegaConf
# import gdown
# from download_url import (
#     get_dataset_download_url,
#     get_normalization_download_url,
#     get_checkpoint_download_url,
# )
import math
import hydra
import sys

import pretty_errors
import logging

from datetime import datetime

# import wandb

# 1. 设置环境变量
@pytest.fixture(scope="module", autouse=True)
def setup_env():
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"


    # wandb.init(disable_telemetry=True)

# 2. 配置 OmegaConf
@pytest.fixture(scope="module", autouse=True)
def setup_omegaconf():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil)
    OmegaConf.register_new_resolver("round_down", math.floor)

    OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))

# 3. 配置日志和输出
@pytest.fixture(scope="module", autouse=True)
def setup_logging():
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    # # Redirect stdout and stderr with line-buffering
    # sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    # sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

    log.debug("Logger initialized")




@pytest.fixture
def cfg():
    """Fixture to load the configuration from YAML."""
    cfg_path = "/home/qtguo/GENERAL/rl/TEST/learning_code/cfg/gym/pretrain/hopper-medium-v2/pre_diffusion_mlp.yaml"
    assert os.path.exists(cfg_path), f"Configuration file not found: {cfg_path}"
    cfg = OmegaConf.load(cfg_path)
    return cfg

def test_pretrain_agent_initialization(cfg):
    """Test initialization of PreTrainAgent with the loaded configuration."""
    agent = PreTrainAgent(cfg)
    assert agent.seed == cfg.get("seed", 42)
    assert agent.n_epochs == cfg.train.n_epochs
    assert agent.batch_size == cfg.train.batch_size
    # The timestamps are changed during the execution
    # assert agent.logdir == cfg.logdir
    print("Initialization test passed!")

def test_pretrain_agent_training_params(cfg):
    """Test if training parameters are correctly set."""
    agent = PreTrainAgent(cfg)

    assert agent.lr_scheduler is not None
    assert agent.optimizer is not None
    print("Training parameter test passed!")

# def test_pretrain_agent_ema(cfg):
#     """Test the EMA component in the agent."""
#     agent = PreTrainAgent(cfg)
#     assert agent.ema.beta == cfg.ema.decay
#     print("EMA component test passed!")

# def test_pretrain_agent_dataset(cfg):
#     """Test if datasets and dataloaders are correctly initialized."""
#     agent = PreTrainAgent(cfg)
#     assert agent.dataset_train is not None
#     assert agent.dataloader_train is not None
#     if "train_split" in cfg.train and cfg.train.train_split < 1:
#         assert agent.dataloader_val is not None
#     print("Dataset initialization test passed!")

# def test_pretrain_agent_run_not_implemented(cfg):
#     """Test if the `run` method raises NotImplementedError."""
#     agent = PreTrainAgent(cfg)
#     with pytest.raises(NotImplementedError):
#         agent.run()
#     print("Run method test passed!")







