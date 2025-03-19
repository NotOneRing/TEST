"""
Pre-training Gaussian/GMM policy

"""

import logging
import wandb
import numpy as np

import tensorflow as tf

log = logging.getLogger(__name__)
from util.timer import Timer

from agent.pretrain.train_agent import PreTrainAgent


from copy import deepcopy


from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, METHOD_NAME


RUN_FUNCTION_TEST_SAVE_LOAD = False


from util.torch_to_tf import torch_tensor_item



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




class TrainGaussianAgent(PreTrainAgent):

    def __init__(self, cfg):
        print("train_gaussian_agent.py: TrainGaussianAgent.__init__()")

        super().__init__(cfg)

        if DEBUG and TEST_LOAD_PRETRAIN:
            self.base_policy_path = cfg.get("base_policy_path", None)

        # Entropy bonus - not used right now since using fixed_std
        self.ent_coef = cfg.train.get("ent_coef", 0)










    def build_ema_model(self, training_flag, item_actions_copy, cond_copy, ent_coef):
        with tf.GradientTape() as tape:
            print("self.model = ", self.model)


            self.ema_model = tf.keras.models.clone_model(self.model)

            if DEBUG:
                self.ema_model.loss_ori_t = None
                self.ema_model.p_losses_noise = None
                self.ema_model.call_noise = None
                self.ema_model.call_noise = None
                self.ema_model.call_x = None
                self.ema_model.q_sample_noise = None


            loss_train_ema = self.ema_model.loss_ori(
                training_flag, 
                item_actions_copy, cond_copy, ent_coef)


            self.reset_parameters()



    def debug_gmm_mlp_save_load(self):

        from tensorflow.keras.utils import get_custom_objects

        from model.common.mlp import MLP, ResidualMLP
        
        cur_dict = {
                "MLP": MLP,
                "ResidualMLP": ResidualMLP, 
        }
        get_custom_objects().update(cur_dict)

        print("self.model.network.mlp_weights = ", self.model.network.mlp_weights)
        for var in self.model.network.mlp_weights.moduleList.variables:
            print(f"1:GMM_MLP.mlp_weights: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")

        import os

        savepath = os.path.join("/ssddata/qtguo/GENERAL_DATA/save_load_test_path/", f"state_{1}.keras")

        tf.keras.models.save_model(self.model.network.mlp_weights, savepath)

        mlp_weights = tf.keras.models.load_model(savepath,  custom_objects=get_custom_objects() )

        
        print("mlp_weights = ", mlp_weights)
        for var in mlp_weights.moduleList.variables:
            print(f"2:GMM_MLP.mlp_weights: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")







    def debug_gmm_save_load(self):

        from tensorflow.keras.utils import get_custom_objects

        from model.common.mlp import MLP, ResidualMLP
        from model.common.mlp_gmm import GMM_MLP
        
        cur_dict = {
                "MLP": MLP,
                "ResidualMLP": ResidualMLP, 
                "GMM_MLP": GMM_MLP
        }
        get_custom_objects().update(cur_dict)

        print("self.model.network = ", self.model.network)
        for var in self.model.network.variables:
            print(f"1:GMM_MLP.network: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")

        num_params = sum(np.prod(var.shape) for var in self.model.network.trainable_variables)
        print(f"1:Number of network parameters: {num_params}")
        print(f"1:Number of network parameters: {sum(var.numpy().size for var in self.model.network.trainable_variables)}")

        import os

        savepath = os.path.join("/ssddata/qtguo/GENERAL_DATA/save_load_test_path/", f"state_{1}.keras")

        tf.keras.models.save_model(self.model.network, savepath)

        network = tf.keras.models.load_model(savepath,  custom_objects=get_custom_objects() )

        num_params = sum(np.prod(var.shape) for var in network.trainable_variables)
        print(f"2:Number of network parameters: {num_params}")
        print(f"2:Number of network parameters: {sum(var.numpy().size for var in network.trainable_variables)}")

        # MLP: call() x.shape =  (16, 4)
        
        print("network = ", network)
        for var in network.variables:
            print(f"2:GMM_MLP.network: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")



    def debug_gmm_load_iter(self):

        from tensorflow.keras.utils import get_custom_objects

        from model.common.mlp import MLP, ResidualMLP
        from model.common.mlp_gmm import GMM_MLP
        
        cur_dict = {
                "MLP": MLP,
                "ResidualMLP": ResidualMLP, 
                "GMM_MLP": GMM_MLP
        }
        get_custom_objects().update(cur_dict)

        print("self.model.network = ", self.model.network)
        for var in self.model.network.variables:
            print(f"1:GMM_MLP.network: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")

        num_params = sum(np.prod(var.shape) for var in self.model.network.trainable_variables)
        print(f"1:Number of network parameters: {num_params}")
        print(f"1:Number of network parameters: {sum(var.numpy().size for var in self.model.network.trainable_variables)}")

        import os

        base_loadpath = "/ssddata/qtguo/GENERAL_DATA/weights_tensorflow/d3il-pretrain/avoid_m3_pre_gmm_mlp_ta4/2025-02-07_03-56-27_42/checkpoint/state_"

        iter_list = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

        for num in iter_list:
            print("iter_list: epoch = ", num)
            loadpath = base_loadpath + str(num) + "_network.keras"
            network = tf.keras.models.load_model(loadpath,  custom_objects=get_custom_objects() )

            num_params = sum(np.prod(var.shape) for var in network.trainable_variables)
            print(f"2:Number of network parameters: {num_params}")
            print(f"2:Number of network parameters: {sum(var.numpy().size for var in network.trainable_variables)}")

            print("network = ", network)
            for var in network.variables:
                print(f"2:GMM_MLP.network: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")











    def debug_gaussian_img_save_load(self):


        if hasattr(self.model.network, 'compress'):
            self.save_load_params("self.model.network.compress", self.model.network.compress)

        if hasattr(self.model.network, 'compress1'):
            self.save_load_params("self.model.network.compress1", self.model.network.compress1)

        if hasattr(self.model.network, 'compress2'):
            self.save_load_params("self.model.network.compress2", self.model.network.compress2)

        if hasattr(self.model.network, 'mlp_logvar'):
            self.save_load_params("self.model.network.logvar", self.model.network.logvar)

        if hasattr(self.model.network, 'mlp_mean'):
            self.save_load_params("self.model.network.mlp_mean", self.model.network.mlp_mean)

        if hasattr(self.model.network, 'backbone'):
            self.save_load_params("self.model.network.backbone", self.model.network.backbone)











    def run(self):

        timer = Timer()
        self.epoch = 1
        cnt_batch = 0
        epoch = 0
        for _ in range(self.n_epochs):
            print("epoch = ", _)
            flag = True

            training_flag = True
            
            loss_train_epoch = []
            ent_train_epoch = []

            if DEBUG and TEST_LOAD_PRETRAIN and _ == 1:
                break

            for batch_train in self.dataloader_train:



                cond = {}
                cur_actions = batch_train['actions']
                cond['state'] = batch_train["states"]

                if 'rgb' in batch_train:
                    cond['rgb'] = batch_train["rgb"]

                item_actions_copy = deepcopy(batch_train['actions'])
                cond_copy = deepcopy(cond)


                if flag:
                    flag = False


                # self.model.train()
                training_flag = True

                with tf.GradientTape() as tape:
                    training_flag=True
                    
                    loss_train, infos_train  = self.model.loss_ori(
                        training_flag, 
                        cur_actions, 
                        cond,
                        ent_coef=self.ent_coef,
                        )



                if epoch == 0:
                    self.build_ema_model(training_flag, item_actions_copy, cond_copy, self.ent_coef)


                if DEBUG and TEST_LOAD_PRETRAIN and epoch == 0:
                    self.load_pickle_network()
                    # break
                

                if DEBUG and TEST_LOAD_PRETRAIN and epoch == 0:
                    self.debug_gaussian_img_save_load()
                    break

                loss_train_epoch.append( torch_tensor_item( loss_train ) )
                ent_train_epoch.append( torch_tensor_item(infos_train["entropy"]) )

                gradients = tape.gradient(loss_train, self.model.trainable_variables)
                zip_gradients_params = zip(gradients, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip_gradients_params)



                # update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
                
                epoch += 1

            loss_train = np.mean(loss_train_epoch)
            ent_train = np.mean(ent_train_epoch)


            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                training_flag = False
                for batch_val in self.dataloader_val:
                    cur_actions = batch_val['actions']
                    cond['state'] = batch_val["states"]
    
                    if 'rgb' in batch_val:
                        cond['rgb'] = batch_val["rgb"]

                    loss_val, infos_val = self.model.loss_ori(training_flag, 
                        cur_actions, cond,
                        ent_coef=self.ent_coef
                        )
                    
                    loss_val_epoch.append( torch_tensor_item(loss_val) )
                training_flag = True
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model(self.epoch)



            # log loss
            if self.epoch % self.log_freq == 0:

                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos_train.items()]
                )
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | {infos_str} | t:{timer():8.4f}"
                )

            # count
            self.epoch += 1





