import os
import random
import numpy as np
import tensorflow as tf

import hydra

from omegaconf import OmegaConf
# import wandb

from copy import deepcopy

from util.torch_to_tf import tf_CosineAnnealingWarmupRestarts, torch_optim_AdamW


# DEBUG = True
# DEBUG = False
from util.config import DEBUG


# DEVICE = "/GPU:0"

# def to_device(x, device=DEVICE):
#     print("train_agent.py: to_device()")

#     if isinstance(x, tf.Tensor):
#         with tf.device(device):
#             return tf.identity(x)
#     elif isinstance(x, dict):
#         return {k: to_device(v, device) for k, v in x.items()}
#     else:
#         print(f"Unrecognized type in `to_device`: {type(x)}")


# def batch_to_device(batch, device="/GPU:0"):
#     print("train_agent.py: batch_to_device()")

#     # 使用 tf.device 来转移数据到指定设备
#     with tf.device(device):
#         # 假设 batch 是一个字典或类似结构，逐个字段转移数据到指定设备
#         vals = {field: to_device(getattr(batch, field), device) for field in batch._fields}
#     return type(batch)(**vals)


class EMA:
    """
    Empirical moving average
    """

    def __init__(self, cfg):
        print("train_agent.py: EMA.__init__()")
        self.beta = cfg.decay

    def update_model_average(self, ma_model, current_model):
        print("train_agent.py: EMA.update_model_average()")
        for ma_weights, current_weights in zip(ma_model.trainable_variables, current_model.trainable_variables):
            ma_weights.assign(self.update_average(ma_weights, current_weights))

    def update_average(self, old, new):
        print("train_agent.py: EMA.update_average()")
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class PreTrainAgent:

    def __init__(self, cfg):
        print("train_agent.py: PreTrainAgent.__init__()")

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        # Set seeds
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        print("cfg = ", cfg)

        print("cfg.model = ", cfg.model)

        # Build model
        # self.model = self.instantiate_model(cfg.model)
        self.model = hydra.utils.instantiate(cfg.model)

        if DEBUG:
            print("train_agent.py: __init__() DEBUG == True")
            self.model.loss_ori_t = None
            self.model.p_losses_noise = None
            self.model.call_noise = None
            self.model.call_noise = None
            self.model.call_x = None
            self.model.q_sample_noise = None


        print("self.model = ", self.model)
        print("Model attributes:", dir(self.model))  # 查看模型的所有属性

        # print("0self.model.loss() = ", self.model.loss())

        print("after instantiate_model")

        self.ema = EMA(cfg.ema)


        # self.ema_model = deepcopy(self.model)
        #把这部分拿到train_diffusion_agent里面去


        print("self.ema = EMA()")

        print("self.model = ", self.model)

        # # 获取 model 的输入形状
        # input_shape = self.model.input_shape

        # # 去掉批次维度 (None)，构建 ema_model
        # self.ema_model.build(input_shape)

        print("after build model")

        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.batch_size = cfg.train.batch_size
        self.epoch_start_ema = cfg.train.get("epoch_start_ema", 20)
        self.update_ema_freq = cfg.train.get("update_ema_freq", 10)
        self.val_freq = cfg.train.get("val_freq", 100)

        print("after training params")

        # Logging, checkpoints
        self.logdir = cfg.logdir

        print("self.logdir = ", self.logdir)

        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq

        print("after logging checkpoints")

        # Build dataset
        self.dataset_train = hydra.utils.instantiate(cfg.train_dataset)
        
        print("after instantiate_dataset()")

        print("type(self.dataset_train) = ", type(self.dataset_train))

        

        if isinstance(self.dataset_train, tf.Tensor):
            print("isinstance(self.dataset_train, tf.Tensor)")
            print(self.dataset_train.shape)



        print("after dataloader_train")



        self.dataloader_val = None

        print("after build dataset")

        self.lr_scheduler = tf_CosineAnnealingWarmupRestarts(
            first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.learning_rate,
            min_lr=cfg.train.lr_scheduler.min_lr,
            warmup_steps=cfg.train.lr_scheduler.warmup_steps,
            gamma=1.0,
        )


        self.optimizer = torch_optim_AdamW(
            self.model.trainable_variables,
            lr=self.lr_scheduler,
            weight_decay=cfg.train.weight_decay,
        )


        # self.reset_parameters()





    def instantiate_model(self, model_cfg):
        print("train_agent.py: instantiate_model()")

        # print("model_cfg = ", model_cfg)
        print("model_cfg = ", model_cfg)
        
        # from model.diffusion.mlp_diffusion import DiffusionMLP

        try:
            model = hydra.utils.instantiate(model_cfg)
        except Exception as e:
            print("Error instantiating model:", e)

        return model
    
        # # Implement model instantiation using tf.keras.Model
        # # Example: return MyModel(**model_cfg)
        # return hydra.utils.instantiate(model_cfg)

    # def instantiate_dataset(self, dataset_cfg):
    #     print("train_agent.py: instantiate_dataset()")
    #     # Implement dataset instantiation
    #     # Example: return MyDataset(**dataset_cfg)
    #     return hydra.utils.instantiate(dataset_cfg)


    def run(self):
        print("train_agent.py: PreTrainAgent.run()")
        raise NotImplementedError



    def reset_parameters(self):
        print("train_agent.py: PreTrainAgent.reset_parameters()")
        self.ema_model.set_weights(self.model.get_weights())



    def step_ema(self):
        print("train_agent.py: PreTrainAgent.step_ema()")
        print("self.epoch_start_ema = ", self.epoch_start_ema)

        if self.epoch < self.epoch_start_ema:
            print("branch self.epoch < self.epoch_start_ema")
            self.reset_parameters()
            return

        self.ema.update_model_average(self.ema_model, self.model)




    def save_model(self, epoch):


        print("train_agent.py: PreTrainAgent.save_model()")

        savepath = os.path.join(self.checkpoint_dir, f"state_{epoch}.keras")
       
        print("savepath = ", savepath)

        # self.model.save(savepath)

        tf.keras.models.save_model(self.model, savepath)

        print(f"Saved model to {savepath}")



        network_savepath = savepath.replace(".keras", "_network.keras")

        print("network_savepath = ", network_savepath)

        # network_config = self.model.network.get_config()
        # print("network_config = ", network_config)

        # self.model.network.save(network_savepath)
        tf.keras.models.save_model(self.model.network, network_savepath)

        print(f"Saved model.network to {network_savepath}")


        # model_network_summary = self.model.network.summary()
        # print("model_network_summary = ", model_network_summary)



        ema_savepath = savepath.replace(".keras", "_ema.keras")

        print("ema_savepath = ", ema_savepath)

        # self.ema_model.save(ema_savepath)
        tf.keras.models.save_model(self.ema_model, ema_savepath)

        print(f"Saved ema_model to {ema_savepath}")


        ema_network_savepath = savepath.replace(".keras", "_ema_network.keras")

        print("ema_network_savepath = ", ema_network_savepath)

        # self.ema_model.network.save(ema_network_savepath)
        tf.keras.models.save_model(self.ema_model.network, ema_network_savepath)

        print(f"Saved ema_model.network to {ema_network_savepath}")


        # ema_model_network_summary = self.ema_model.network.summary()
        # print("ema_model_network_summary = ", ema_model_network_summary)






        # network_mlpmean_savepath = savepath.replace(".keras", "_network_mlpmean.keras")

        # print("network_mlpmean_savepath = ", network_mlpmean_savepath)

        # self.model.network.mlp_mean.save(network_mlpmean_savepath)

        # print(f"Saved model.network.mlp_mean to {network_mlpmean_savepath}")




        # ema_network_mlpmean_savepath = savepath.replace(".keras", "_ema_network_mlpmean.keras")

        # print("ema_network_mlpmean_savepath = ", ema_network_mlpmean_savepath)

        # self.ema_model.network.mlp_mean.save(ema_network_mlpmean_savepath)

        # print(f"Saved ema_model.network.mlp_mean to {ema_network_mlpmean_savepath}")





        # network_condmlp_savepath = savepath.replace(".keras", "_network_condmlp.keras")

        # print("network_condmlp_savepath = ", network_condmlp_savepath)

        # self.model.network.cond_mlp.save(network_condmlp_savepath)

        # print(f"Saved model.network.cond_mlp to {network_condmlp_savepath}")




        # ema_network_condmlp_savepath = savepath.replace(".keras", "_ema_network_condmlp.keras")

        # print("ema_network_condmlp_savepath = ", ema_network_condmlp_savepath)

        # self.ema_model.network.cond_mlp.save(ema_network_condmlp_savepath)

        # print(f"Saved ema_model.network.cond_mlp to {ema_network_condmlp_savepath}")











        # network_timeemb_savepath = savepath.replace(".keras", "_network_timeemb.keras")

        # print("network_timeemb_savepath = ", network_timeemb_savepath)

        # self.model.network.time_embedding.save(network_timeemb_savepath)

        # print(f"Saved model.network.time_embedding to {network_timeemb_savepath}")




        # ema_network_timeemb_savepath = savepath.replace(".keras", "_ema_network_timeemb.keras")

        # print("ema_network_timeemb_savepath = ", ema_network_timeemb_savepath)

        # self.ema_model.network.time_embedding.save(ema_network_timeemb_savepath)

        # print(f"Saved ema_model.network.time_embedding to {ema_network_timeemb_savepath}")





    def load(self, epoch):
        print("train_agent.py: PreTrainAgent.load()")
        # loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.h5")
        loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.keras")

        print("loadpath = ", loadpath)

        from model.diffusion.mlp_diffusion import DiffusionMLP
        from model.diffusion.diffusion import DiffusionModel
        from model.common.mlp import MLP, ResidualMLP, TwoLayerPreActivationResNetLinear
        from model.diffusion.modules import SinusoidalPosEmb
        from model.common.modules import SpatialEmb, RandomShiftsAug
        from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish, nn_Identity

        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
            'DiffusionMLP': DiffusionMLP,
            # 'VPGDiffusion': VPGDiffusion,
            'SinusoidalPosEmb': SinusoidalPosEmb,  # 假设 SinusoidalPosEmb 是你自定义的层
            'MLP': MLP,                            # 自定义的 MLP 层
            'ResidualMLP': ResidualMLP,            # 自定义的 ResidualMLP 层
            'nn_Sequential': nn_Sequential,        # 自定义的 Sequential 类
            "nn_Identity": nn_Identity,
            'nn_Linear': nn_Linear,
            'nn_LayerNorm': nn_LayerNorm,
            'nn_Dropout': nn_Dropout,
            'nn_ReLU': nn_ReLU,
            'nn_Mish': nn_Mish,
            'SpatialEmb': SpatialEmb,
            'RandomShiftsAug': RandomShiftsAug,
            "TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
         }
        # Register your custom class with Keras
        get_custom_objects().update(cur_dict)





        self.model = tf.keras.models.load_model(loadpath,  custom_objects=get_custom_objects() )

        # self.model = self.model2


        self.model.network = tf.keras.models.load_model(loadpath.replace(".keras", "_network.keras") ,  custom_objects=get_custom_objects() )

        # self.model.network.cond_mlp = tf.keras.models.load_model(loadpath.replace(".keras", "_network_condmlp.keras") ,  custom_objects=get_custom_objects() )
        # self.model.network.mlp_mean = tf.keras.models.load_model(loadpath.replace(".keras", "_network_mlpmean.keras") ,  custom_objects=get_custom_objects() )
        # self.model.network.time_embedding = tf.keras.models.load_model(loadpath.replace(".keras", "_network_timeemb.keras") ,  custom_objects=get_custom_objects() )


        # model_network_summary = self.model.network.summary()
        # print("model_network_summary = ", model_network_summary)


        # self.model.network.mlp_mean = tf.keras.models.load_model(loadpath.replace(".keras", "_network_mlpmean.keras"))


        # self.ema_model = tf.keras.models.load_model(loadpath.replace(".keras", "_ema.keras"))
        self.ema_model = tf.keras.models.load_model(loadpath.replace(".keras", "_ema.keras"),  custom_objects=get_custom_objects() )

        # self.ema_model = self.ema_model2

        self.ema_model.network = tf.keras.models.load_model(loadpath.replace(".keras", "_ema_network.keras") ,  custom_objects=get_custom_objects() )

        # self.ema_model.network.cond_mlp = tf.keras.models.load_model(loadpath.replace(".keras", "_ema_network_condmlp.keras") ,  custom_objects=get_custom_objects() )
        # self.ema_model.network.mlp_mean = tf.keras.models.load_model(loadpath.replace(".keras", "_ema_network_mlpmean.keras") ,  custom_objects=get_custom_objects() )
        # self.ema_model.network.time_embedding = tf.keras.models.load_model(loadpath.replace(".keras", "_ema_network_timeemb.keras") ,  custom_objects=get_custom_objects() )

        # ema_model_network_summary = self.ema_model.network.summary()
        # print("ema_model_network_summary = ", ema_model_network_summary)


        # self.ema_model.network.mlp_mean = tf.keras.models.load_model(loadpath.replace(".keras", "_ema_network_mlpmean.keras"))
















