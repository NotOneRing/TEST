import os
import random
import numpy as np
import tensorflow as tf

import hydra

from omegaconf import OmegaConf
import wandb
from copy import deepcopy

DEVICE = "/GPU:0"

def to_device(x, device=DEVICE):
    print("train_agent.py: to_device()", flush=True)

    if isinstance(x, tf.Tensor):
        with tf.device(device):
            return tf.identity(x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")


def batch_to_device(batch, device="/GPU:0"):
    print("train_agent.py: batch_to_device()", flush=True)

    # 使用 tf.device 来转移数据到指定设备
    with tf.device(device):
        # 假设 batch 是一个字典或类似结构，逐个字段转移数据到指定设备
        vals = {field: to_device(getattr(batch, field), device) for field in batch._fields}
    return type(batch)(**vals)


class EMA:
    """
    Empirical moving average
    """

    def __init__(self, cfg):
        print("train_agent.py: EMA.__init__()", flush=True)
        self.beta = cfg.decay

    def update_model_average(self, ma_model, current_model):
        print("train_agent.py: EMA.update_model_average()", flush=True)
        for ma_weights, current_weights in zip(ma_model.trainable_variables, current_model.trainable_variables):
            ma_weights.assign(self.update_average(ma_weights, current_weights))

    def update_average(self, old, new):
        print("train_agent.py: EMA.update_average()", flush=True)
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class PreTrainAgent:

    def __init__(self, cfg):
        print("train_agent.py: PreTrainAgent.__init__()", flush=True)

        # Set seeds
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Wandb
        self.use_wandb = cfg.wandb is not None
        if self.use_wandb:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Build model
        self.model = self.instantiate_model(cfg.model)

        print("after instantiate_model", flush = True)

        self.ema = EMA(cfg.ema)

        print("self.ema = EMA()", flush = True)

        self.ema_model = deepcopy(self.model)

        # # 获取 model 的输入形状
        # input_shape = self.model.input_shape

        # # 去掉批次维度 (None)，构建 ema_model
        # self.ema_model.build(input_shape)

        print("after build model", flush=True)

        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.batch_size = cfg.train.batch_size
        self.epoch_start_ema = cfg.train.get("epoch_start_ema", 20)
        self.update_ema_freq = cfg.train.get("update_ema_freq", 10)
        self.val_freq = cfg.train.get("val_freq", 100)

        print("after training params", flush=True)

        # Logging, checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq

        print("after logging checkpoints", flush=True)

        # Build dataset
        self.dataset_train = self.instantiate_dataset(cfg.train_dataset)
        
        print("after instantiate_dataset()", flush=True)

        print(type(self.dataset_train))
        if isinstance(self.dataset_train, tf.Tensor):
            print(self.dataset_train.shape)

        self.dataloader_train = (
            tf.data.Dataset.from_generator(
                lambda: (self.dataset_train[i] for i in range(len(self.dataset_train))),
                output_signature=self.dataset_train.element_spec,
            )
            .shuffle(1000)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # 打印生成的数据的形状和内容，验证是否正确
        for batch in self.dataloader_train.take(1):
            print("Actions shape:", batch[0].shape)
            print("Conditions shape:", {k: v.shape for k, v in batch[1].items()})


        print("after dataloader_train", flush=True)

        self.dataloader_val = None

        print("after build dataset", flush=True)

        # Optimizer and scheduler
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=cfg.train.learning_rate,
                first_decay_steps=cfg.train.lr_scheduler.first_cycle_steps,
                t_mul=1.0,
                alpha=cfg.train.lr_scheduler.min_lr / cfg.train.learning_rate,
            ),
        )

        # print("self.batch_size = ", self.batch_size)

        # # 使用 `take(1)` 仅获取第一个批次并获取其形状
        # sample_input = next(iter(self.dataloader_train.take(1)))  # 获取第一个batch的数据
        # input_shape = sample_input[0].shape  # 获取输入特征的形状

        # print(f"Input shape: {input_shape}")
        # # 假设输入是一个张量，初始化一个 dummy variable
        # dummy_input = tf.zeros(*input_shape)  # 创建一个 shape 为 (input_shape) 的张量

        # print("before model", flush=True)
        # # # 初始化模型
        # # self.model.build(dummy_input.shape)
        # # self.ema_model.build(dummy_input.shape)
        # self.model(dummy_input)  # 这会触发权重的创建
        # self.ema_model(dummy_input)  # 同样触发 ema_model 的权重创建

        # print("after optimizer", flush=True)

        # self.reset_parameters()

        # print("after reset_parameters()", flush=True)

    def instantiate_model(self, model_cfg):
        print("train_agent.py: instantiate_model()", flush=True)

        print(" ", model_cfg, flush=True)

        try:
            model = hydra.utils.instantiate(model_cfg)
        except Exception as e:
            print("Error instantiating model:", e)

        return model
        # # Implement model instantiation using tf.keras.Model
        # # Example: return MyModel(**model_cfg)
        # return hydra.utils.instantiate(model_cfg)

    def instantiate_dataset(self, dataset_cfg):
        print("train_agent.py: instantiate_dataset()", flush=True)
        # Implement dataset instantiation
        # Example: return MyDataset(**dataset_cfg)
        return hydra.utils.instantiate(dataset_cfg)

    def run(self):
        print("train_agent.py: PreTrainAgent.run()", flush=True)
        raise NotImplementedError

    def reset_parameters(self):
        print("train_agent.py: PreTrainAgent.reset_parameters()", flush=True)
        self.ema_model.set_weights(self.model.get_weights())

    def step_ema(self):
        print("train_agent.py: PreTrainAgent.step_ema()", flush=True)
        if self.epoch < self.epoch_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)


    def save_model(self):
        print("train_agent.py: PreTrainAgent.save_model()", flush=True)
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.h5")
        self.model.save_weights(savepath)
        self.ema_model.save_weights(savepath.replace(".h5", "_ema.h5"))
        print(f"Saved model to {savepath}")


    def load(self, epoch):
        print("train_agent.py: PreTrainAgent.load()", flush=True)
        loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.h5")
        self.model.load_weights(loadpath)
        self.ema_model.load_weights(loadpath.replace(".h5", "_ema.h5"))




