import os
import random
import numpy as np
import tensorflow as tf

import hydra

from omegaconf import OmegaConf

from copy import deepcopy

from util.torch_to_tf import CosineAWR,\
      torch_optim_AdamW, torch_utils_data_DataLoader


from util.config import DEBUG, METHOD_NAME


from util.torch_to_tf import nn_TransformerEncoder, nn_TransformerEncoderLayer, nn_TransformerDecoder,\
nn_TransformerDecoderLayer, einops_layers_torch_Rearrange, nn_GroupNorm, nn_ConvTranspose1d, nn_Conv2d, nn_Conv1d, \
nn_MultiheadAttention, nn_LayerNorm, nn_Embedding, nn_ModuleList, nn_Sequential, \
nn_Linear, nn_Dropout, nn_ReLU, nn_GELU, nn_ELU, nn_Mish, nn_Softplus, nn_Identity, nn_Tanh
from model.diffusion.unet import ResidualBlock1D, Unet1D
from model.diffusion.modules import Conv1dBlock, Upsample1d, Downsample1d, SinusoidalPosEmb
from model.common.vit import VitEncoder, PatchEmbed1, PatchEmbed2, MultiHeadAttention, TransformerLayer, MinVit
from model.common.transformer import Gaussian_Transformer, GMM_Transformer, Transformer
from model.common.modules import SpatialEmb, RandomShiftsAug
from model.common.mlp import MLP, ResidualMLP, TwoLayerPreActivationResNetLinear
from model.common.mlp_gaussian import Gaussian_MLP, Gaussian_VisionMLP


cur_dict = {
#part1:
"nn_TransformerEncoder": nn_TransformerEncoder, 
"nn_TransformerEncoderLayer": nn_TransformerEncoderLayer, 
"nn_TransformerDecoder": nn_TransformerDecoder,
"nn_TransformerDecoderLayer": nn_TransformerDecoderLayer, 
"einops_layers_torch_Rearrange": einops_layers_torch_Rearrange, 
"nn_GroupNorm": nn_GroupNorm, 
"nn_ConvTranspose1d": nn_ConvTranspose1d, 
"nn_Conv2d": nn_Conv2d, 
"nn_Conv1d": nn_Conv1d,
"nn_MultiheadAttention": nn_MultiheadAttention,
"nn_LayerNorm": nn_LayerNorm, 
"nn_Embedding": nn_Embedding, 
"nn_ModuleList": nn_ModuleList, 
"nn_Sequential": nn_Sequential,
"nn_Linear": nn_Linear, 
"nn_Dropout": nn_Dropout, 
"nn_ReLU": nn_ReLU, 
"nn_GELU": nn_GELU, 
"nn_ELU": nn_ELU, 
"nn_Mish": nn_Mish, 
"nn_Softplus": nn_Softplus, 
"nn_Identity": nn_Identity, 
"nn_Tanh": nn_Tanh,
#part2:
"ResidualBlock1D": ResidualBlock1D,
"Unet1D": Unet1D,
"Conv1dBlock": Conv1dBlock, 
"Upsample1d": Upsample1d, 
"Downsample1d": Downsample1d, 
"SinusoidalPosEmb": SinusoidalPosEmb,
#part3:
"VitEncoder": VitEncoder, 
"PatchEmbed1": PatchEmbed1, 
"PatchEmbed2": PatchEmbed2,
"MultiHeadAttention": MultiHeadAttention, 
"TransformerLayer": TransformerLayer, 
"MinVit": MinVit,
"Gaussian_Transformer": Gaussian_Transformer, 
"GMM_Transformer": GMM_Transformer, 
"Transformer": Transformer,
"SpatialEmb": SpatialEmb,
"RandomShiftsAug": RandomShiftsAug,
"MLP": MLP,
"ResidualMLP": ResidualMLP, 
"TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
"Gaussian_MLP": Gaussian_MLP, 
"Gaussian_VisionMLP": Gaussian_VisionMLP,
}


class EMA:
    """
    Empirical moving average
    """

    def __init__(self, cfg):
        print("train_agent.py: EMA.__init__()")
        self.beta = cfg.decay

    def update_model_average(self, ma_model, current_model):
        print("train_agent.py: EMA.update_model_average()")



        if DEBUG:


            epoch0_model_weights = current_model.get_weights()

            epoch0_model_network_weights = current_model.network.get_weights()

            epoch0_model_network_mlpmean_weights = current_model.network.mlp_mean.get_weights()

            if current_model.network.cond_mlp:
                epoch0_model_network_condmlp_weights = current_model.network.cond_mlp.get_weights()


            epoch0_model_network_timeemb_weights = current_model.network.time_embedding.get_weights()








            epoch0_model_trainable_variables = deepcopy(current_model.trainable_variables)

            epoch0_model_network_trainable_variables = deepcopy(current_model.network.trainable_variables)

            epoch0_model_network_mlpmean_trainable_variables = deepcopy(current_model.network.mlp_mean.trainable_variables)

            if current_model.network.cond_mlp:
                epoch0_model_network_condmlp_trainable_variables = deepcopy(current_model.network.cond_mlp.trainable_variables)

            epoch0_model_network_timeemb_trainable_variables = deepcopy(current_model.network.time_embedding.trainable_variables)










            epoch0_ema_model_weights = ma_model.get_weights()

            epoch0_ema_model_network_weights = ma_model.network.get_weights()

            epoch0_ema_model_network_mlpmean_weights = ma_model.network.mlp_mean.get_weights()

            if ma_model.network.cond_mlp:
                epoch0_ema_model_network_condmlp_weights = ma_model.network.cond_mlp.get_weights()


            epoch0_ema_model_network_timeemb_weights = ma_model.network.time_embedding.get_weights()





            epoch0_ema_model_trainable_variables = deepcopy( ma_model.trainable_variables )

            epoch0_ema_model_network_trainable_variables = deepcopy( ma_model.network.trainable_variables )

            epoch0_ema_model_network_mlpmean_trainable_variables = deepcopy( ma_model.network.mlp_mean.trainable_variables )

            if ma_model.network.cond_mlp:
                epoch0_ema_model_network_condmlp_trainable_variables = deepcopy( ma_model.network.cond_mlp.trainable_variables )

            epoch0_ema_model_network_timeemb_trainable_variables = deepcopy( ma_model.network.time_embedding.trainable_variables )


        for ma_weights, current_weights in zip(ma_model.trainable_variables, current_model.trainable_variables):
            updated_parameters = self.update_average(ma_weights, current_weights)
            ma_weights.assign(updated_parameters)



        if DEBUG:


            epoch_model_weights = ma_model.get_weights()

            for i in range(len(epoch_model_weights)):
                part1_weights = epoch0_model_weights[i] * (1-self.beta) + epoch0_ema_model_weights[i] * self.beta
                part2_weights = epoch_model_weights[i]
                assert np.allclose(part1_weights, part2_weights), "np.allclose(epoch0_model_weights[i] - epoch_model_weights[i]) != 0"


            epoch_model_trainable_variables = ma_model.trainable_variables
            for i in range(len(epoch_model_trainable_variables)):
                part1 = (epoch0_model_trainable_variables[i] * (1-self.beta) + epoch0_ema_model_trainable_variables[i] * self.beta).numpy()
                part2 = (epoch_model_trainable_variables[i]).numpy()
                assert np.allclose(part1, part2, atol=1e-4), "np.allclose(epoch0_model_trainable_variables[i] - epoch_model_trainable_variables[i]) != 0"









            epoch_model_network_weights = ma_model.network.get_weights()

            for i in range(len(epoch_model_network_weights)):
                part1_weights = epoch0_model_network_weights[i] * (1-self.beta) + epoch0_ema_model_network_weights[i] * self.beta
                part2_weights = epoch_model_network_weights[i]
                assert np.allclose(part1_weights, part2_weights ), "np.allclose(epoch0_model_network_weights[i] - epoch_model_network_weights[i]) != 0"

            epoch_model_network_trainable_variables = ma_model.network.trainable_variables
            for i in range(len(epoch_model_network_trainable_variables)):
                part1 = (epoch0_model_network_trainable_variables[i] * (1-self.beta) + epoch0_ema_model_network_trainable_variables[i] * self.beta).numpy()
                part2 = (epoch_model_network_trainable_variables[i]).numpy()
                assert np.allclose(part1, part2 , atol=1e-4), "np.allclose(epoch0_model_network_trainable_variables[i] - epoch_model_network_trainable_variables[i]) != 0"







            epoch_model_network_mlpmean_weights = ma_model.network.mlp_mean.get_weights()

            for i in range(len(epoch_model_network_mlpmean_weights)):
                part1_weights = epoch0_model_network_mlpmean_weights[i] * (1-self.beta) + epoch0_ema_model_network_mlpmean_weights[i] * self.beta
                part2_weights = epoch_model_network_mlpmean_weights[i]
                assert np.allclose(part1_weights, part2_weights ), "np.allclose(epoch0_model_network_mlpmean_weights[i] - epoch_model_network_mlpmean_weights[i]) != 0"


            epoch_model_network_mlpmean_trainable_variables = ma_model.network.mlp_mean.trainable_variables
            for i in range(len(epoch_model_network_mlpmean_trainable_variables)):
                part1 = (epoch0_model_network_mlpmean_trainable_variables[i] * (1-self.beta) + epoch0_ema_model_network_mlpmean_trainable_variables[i] * self.beta).numpy()
                part2 = (epoch_model_network_mlpmean_trainable_variables[i]).numpy()
                assert np.allclose(part1, part2 , atol=1e-4), "np.allclose(epoch0_model_network_mlpmean_trainable_variables[i] - epoch_model_network_mlpmean_trainable_variables[i]) != 0"








            if ma_model.network.cond_mlp:

                epoch_model_network_condmlp_weights = ma_model.network.cond_mlp.get_weights()

                for i in range(len(epoch_model_network_condmlp_weights)):
                    part1_weights = epoch0_model_network_condmlp_weights[i] * (1-self.beta) + epoch0_ema_model_network_condmlp_weights[i] * self.beta
                    part2_weights = epoch_model_network_condmlp_weights[i]
                    assert np.allclose(part1_weights, part2_weights), "np.allclose(epoch0_model_network_condmlp_weights[i] - epoch_model_network_condmlp_weights[i]) != 0"

            if ma_model.network.cond_mlp:

                epoch_model_network_condmlp_trainable_variables = ma_model.network.cond_mlp.trainable_variables
                for i in range(len(epoch_model_network_condmlp_trainable_variables)):
                    part1 = (epoch0_model_network_condmlp_trainable_variables[i] * (1-self.beta) + epoch0_ema_model_network_condmlp_trainable_variables[i] * self.beta).numpy()
                    part2 = (epoch_model_network_condmlp_trainable_variables[i]).numpy()
                    assert np.allclose(part1, part2 , atol=1e-4), "np.allclose(epoch0_model_network_condmlp_trainable_variables[i] - epoch_model_network_condmlp_trainable_variables[i]) != 0"





            epoch_model_network_timeemb_weights = ma_model.network.time_embedding.get_weights()

            for i in range(len(epoch_model_network_timeemb_weights)):
                part1_weights = epoch0_model_network_timeemb_weights[i] * (1-self.beta) + epoch0_ema_model_network_timeemb_weights[i] * self.beta
                part2_weights = epoch_model_network_timeemb_weights[i]
                assert np.allclose(part1_weights, part2_weights ), "np.allclose(epoch0_model_network_timeemb_weights[i] - epoch_model_network_timeemb_weights[i]) != 0"


            epoch_model_network_timeemb_trainable_variables = ma_model.network.time_embedding.trainable_variables
            for i in range(len(epoch_model_network_timeemb_trainable_variables)):
                part1 = (epoch0_model_network_timeemb_trainable_variables[i] * (1-self.beta) + epoch0_ema_model_network_timeemb_trainable_variables[i] * self.beta).numpy()
                part2 = (epoch_model_network_timeemb_trainable_variables[i]).numpy()
                assert np.allclose(part1, part2 , atol=1e-4), "np.allclose(epoch0_model_network_timeemb_trainable_variables[i] - epoch_model_network_timeemb_trainable_variables[i]) != 0"









        

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

        self.cfg_env_name = cfg.get("env_name", None)

        if self.cfg_env_name is None:
            self.cfg_env_name = cfg.env
        

        self.model = hydra.utils.instantiate(cfg.model, env_name=self.cfg_env_name)


        if DEBUG:
            print("train_agent.py: __init__() DEBUG == True")
            self.model.loss_ori_t = None
            self.model.p_losses_noise = None
            self.model.call_noise = None
            self.model.call_noise = None
            self.model.call_x = None
            self.model.q_sample_noise = None



        self.ema = EMA(cfg.ema)


        print("self.ema = EMA()")

        print("self.model = ", self.model)

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

        print("type(self.dataset_train) = ", self.dataset_train)


        self.dataloader_train = torch_utils_data_DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=4 if self.dataset_train.device == "cpu" else 0,
            shuffle=True,
            pin_memory=True if self.dataset_train.device == "cpu" else False,
        )


        print("after instantiate_dataset()")

        print("type(self.dataset_train) = ", type(self.dataset_train))

        

        if isinstance(self.dataset_train, tf.Tensor):
            print("isinstance(self.dataset_train, tf.Tensor)")
            print(self.dataset_train.shape)



        print("after dataloader_train")



        self.dataloader_val = None

        print("after build dataset")

        self.lr_scheduler = CosineAWR(
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






    def instantiate_model(self, model_cfg):
        print("train_agent.py: instantiate_model()")

        print("model_cfg = ", model_cfg)
        
        try:
            model = hydra.utils.instantiate(model_cfg)
        except Exception as e:
            print("Error instantiating model:", e)

        return model


    def run(self):
        print("train_agent.py: PreTrainAgent.run()")
        raise NotImplementedError



    def reset_parameters(self):
        print("train_agent.py: PreTrainAgent.reset_parameters()")
        self.ema_model.set_weights(self.model.get_weights())



    def step_ema(self):
        print("train_agent.py: PreTrainAgent.step_ema()")

        print("self.epoch = ", self.epoch)
        print("self.epoch_start_ema = ", self.epoch_start_ema)

        if self.epoch < self.epoch_start_ema:
            print("branch self.epoch < self.epoch_start_ema")
            self.reset_parameters()
            return

        print("self.ema.update_model_average(self.ema_model, self.model)")

        self.ema.update_model_average(self.ema_model, self.model)





    def load_load_pretrain_model(self, loadpath):

        print("train_agent.py: PreTrainAgent.load_load_pretrain_model()")

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
            'SinusoidalPosEmb': SinusoidalPosEmb,   
            'MLP': MLP,                            # Custom MLP layer
            'ResidualMLP': ResidualMLP,            # Custom ResidualMLP layer
            'nn_Sequential': nn_Sequential,        # Custom Sequential class
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
        # Register custom class with Keras
        get_custom_objects().update(cur_dict)

        print("before first model load")
        self.model = tf.keras.models.load_model(loadpath,  custom_objects=get_custom_objects() )

        print("before network model load")
        self.model.network = tf.keras.models.load_model(loadpath.replace(".keras", "_network.keras") ,  custom_objects=get_custom_objects() )

        print("after all model loaded")









    def save_load_pretrain_model(self, savepath):


        print("train_agent.py: PreTrainAgent.save_load_pretrain_model()")

       
        print("savepath = ", savepath)

        tf.keras.models.save_model(self.model, savepath)

        print(f"Saved model to {savepath}")

        network_savepath = savepath.replace(".keras", "_network.keras")

        print("network_savepath = ", network_savepath)

        tf.keras.models.save_model(self.model.network, network_savepath)

        print(f"Saved model.network to {network_savepath}")




    def save_model(self, epoch):


        print("train_agent.py: PreTrainAgent.save_model()")

        savepath = os.path.join(self.checkpoint_dir, f"state_{epoch}.keras")
       
        print("savepath = ", savepath)

        tf.keras.models.save_model(self.model, savepath)

        print(f"Saved model to {savepath}")


        network_savepath = savepath.replace(".keras", "_network.keras")

        print("network_savepath = ", network_savepath)

        tf.keras.models.save_model(self.model.network, network_savepath)

        print(f"Saved model.network to {network_savepath}")



        ema_savepath = savepath.replace(".keras", "_ema.keras")

        print("ema_savepath = ", ema_savepath)

        tf.keras.models.save_model(self.ema_model, ema_savepath)

        print(f"Saved ema_model to {ema_savepath}")


        ema_network_savepath = savepath.replace(".keras", "_ema_network.keras")

        print("ema_network_savepath = ", ema_network_savepath)

        tf.keras.models.save_model(self.ema_model.network, ema_network_savepath)

        print(f"Saved ema_model.network to {ema_network_savepath}")




    def load(self, epoch):
        print("train_agent.py: PreTrainAgent.load()")
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
            'SinusoidalPosEmb': SinusoidalPosEmb,   
            'MLP': MLP,                            # Custom MLP layer
            'ResidualMLP': ResidualMLP,            # Custom ResidualMLP layer
            'nn_Sequential': nn_Sequential,        # Custom Sequential class
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
        # Register custom class with Keras
        get_custom_objects().update(cur_dict)





        self.model = tf.keras.models.load_model(loadpath,  custom_objects=get_custom_objects() )

        self.ema_model = tf.keras.models.load_model(loadpath.replace(".keras", "_ema.keras"),  custom_objects=get_custom_objects() )




    def save_model_test(self, epoch):
        print("train_agent.py: PreTrainAgent.save_model()")

        savepath = os.path.join("/ssddata/qtguo/GENERAL_DATA/save_load_test_path/", f"state_{epoch}.keras")
       
        print("savepath = ", savepath)

        tf.keras.models.save_model(self.model, savepath)

        print(f"Saved model to {savepath}")


        network_savepath = savepath.replace(".keras", "_network.keras")

        print("network_savepath = ", network_savepath)

        tf.keras.models.save_model(self.model.network, network_savepath)

        print(f"Saved model.network to {network_savepath}")


        ema_savepath = savepath.replace(".keras", "_ema.keras")

        print("ema_savepath = ", ema_savepath)

        tf.keras.models.save_model(self.ema_model, ema_savepath)

        print(f"Saved ema_model to {ema_savepath}")


        ema_network_savepath = savepath.replace(".keras", "_ema_network.keras")

        print("ema_network_savepath = ", ema_network_savepath)

        tf.keras.models.save_model(self.ema_model.network, ema_network_savepath)

        print(f"Saved ema_model.network to {ema_network_savepath}")





    def load_model_test(self, epoch):
        print("train_agent.py: PreTrainAgent.load()")
        loadpath = os.path.join("/ssddata/qtguo/GENERAL_DATA/save_load_test_path/", f"state_{epoch}.keras")

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
            'SinusoidalPosEmb': SinusoidalPosEmb,   
            'MLP': MLP,                            # Custom MLP layer
            'ResidualMLP': ResidualMLP,            # Custom ResidualMLP layer
            'nn_Sequential': nn_Sequential,        # Custom Sequential class
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
        # Register custom class with Keras
        get_custom_objects().update(cur_dict)





        self.model = tf.keras.models.load_model(loadpath,  custom_objects=get_custom_objects() )


        self.ema_model = tf.keras.models.load_model(loadpath.replace(".keras", "_ema.keras"),  custom_objects=get_custom_objects() )

































    def load_pickle_network(self):            
        if METHOD_NAME == "Diffusion_MLP":
            print("self.model.load_pickle()")
            self.model.load_pickle(self.base_policy_path)
        elif METHOD_NAME == "Diffusion_UNet":
            print("self.model.load_pickle_diffusion_unet()")
            self.model.load_pickle_diffusion_unet(self.base_policy_path)
        elif METHOD_NAME == "Diffusion_ViT_UNet":
            print("self.model.load_pickle_diffusion_unet_img()")
            self.model.load_pickle_diffusion_unet_img(self.base_policy_path)
        elif METHOD_NAME == "Gaussian_MLP":
            print("self.model.load_pickle_gaussian_mlp()")
            self.model.load_pickle_gaussian_mlp(self.base_policy_path)
        elif METHOD_NAME == "GMM_MLP":
            print("self.model.load_pickle_gmm_mlp()")
            self.model.load_pickle_gmm_mlp(self.base_policy_path)
        elif METHOD_NAME == "Gaussian_ViT_MLP":
            print("self.model.load_pickle_gaussian_mlp_img()")
            self.model.load_pickle_gaussian_mlp_img(self.base_policy_path)
        elif METHOD_NAME == "Diffusion_ViT_MLP":
            self.model.load_pickle_diffusion_mlp_img(self.base_policy_path)
        else:
            raise RuntimeError("Method Undefined")

        savepath = self.base_policy_path.replace(".pt", ".keras")


        if "ViT" in METHOD_NAME:            
            self.model.build_actor_vision(self.model.network)
        else:
            self.model.build_actor(self.model.network)


        self.save_load_pretrain_model(savepath)






    def save_load_params(self, params_name, params):

        print("params_name =", params_name)
        print("params = ", params)

        for var in params.variables:
            print(f"1: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")

        num_params = sum(np.prod(var.shape) for var in params.trainable_variables)
        print(f"1:Number of network parameters: {num_params}")
        print(f"1:Number of network parameters: {sum(var.numpy().size for var in params.trainable_variables)}")

        import os

        savepath = os.path.join("/ssddata/qtguo/GENERAL_DATA/save_load_test_path/", f"state_{1}.keras")


        tf.keras.models.save_model(params, savepath)


        from tensorflow.keras.utils import get_custom_objects


        get_custom_objects().update(cur_dict)

        network = tf.keras.models.load_model(savepath, custom_objects=get_custom_objects() )

        for var in network.variables:
            print("2:", params_name, f": Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")


        num_params = sum(np.prod(var.shape) for var in network.trainable_variables)
        print(f"2:Number of network parameters: {num_params}")
        print(f"2:Number of network parameters: {sum(var.numpy().size for var in network.trainable_variables)}")









