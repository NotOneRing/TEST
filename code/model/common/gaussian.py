"""
Gaussian policy parameterization.

"""


import tensorflow as tf

import numpy as np

import logging

log = logging.getLogger(__name__)

from util.torch_to_tf import Normal, torch_ones_like, torch_clamp, torch_mean, torch_tensor_view,\
torch_log, torch_tanh, torch_sum, nn_Parameter, torch_tensor, torch_ones




from util.config import OUTPUT_VARIABLES, OUTPUT_FUNCTION_HEADER, OUTPUT_POSITIONS, METHOD_NAME


from util.torch_to_tf import nn_TransformerEncoder, nn_TransformerEncoderLayer, nn_TransformerDecoder,\
nn_TransformerDecoderLayer, einops_layers_torch_Rearrange, nn_GroupNorm, nn_ConvTranspose1d, nn_Conv2d, nn_Conv1d, \
nn_MultiheadAttention, nn_LayerNorm, nn_Embedding, nn_ModuleList, nn_Sequential, \
nn_Linear, nn_Dropout, nn_ReLU, nn_GELU, nn_ELU, nn_Mish, nn_Softplus, nn_Identity, nn_Tanh
# from model.rl.gaussian_calql import CalQL_Gaussian
from model.diffusion.unet import ResidualBlock1D, Unet1D
from model.diffusion.modules import Conv1dBlock, Upsample1d, Downsample1d, SinusoidalPosEmb
# from model.diffusion.mlp_diffusion import DiffusionMLP, VisionDiffusionMLP
# from model.diffusion.eta import EtaStateAction, EtaState, EtaAction, EtaFixed
# from model.diffusion.diffusion import DiffusionModel
from model.common.vit import VitEncoder, PatchEmbed1, PatchEmbed2, MultiHeadAttention, TransformerLayer, MinVit
from model.common.transformer import Gaussian_Transformer, GMM_Transformer, Transformer
from model.common.modules import SpatialEmb, RandomShiftsAug
from model.common.mlp import MLP, ResidualMLP, TwoLayerPreActivationResNetLinear
# from model.common.mlp_gmm import GMM_MLP
from model.common.mlp_gaussian import Gaussian_MLP, Gaussian_VisionMLP
# from model.common.gaussian import  GaussianModel
# from model.common.critic import CriticObs, CriticObsAct
# from model.common.gmm import GMMModel


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
# "CalQL_Gaussian": CalQL_Gaussian,
"ResidualBlock1D": ResidualBlock1D,
"Unet1D": Unet1D,
"Conv1dBlock": Conv1dBlock, 
"Upsample1d": Upsample1d, 
"Downsample1d": Downsample1d, 
"SinusoidalPosEmb": SinusoidalPosEmb,
# "DiffusionMLP": DiffusionMLP, 
# # "VisionDiffusionMLP": VisionDiffusionMLP,
# "EtaStateAction": EtaStateAction, 
# "EtaState": EtaState, 
# "EtaAction": EtaAction, 
# "EtaFixed": EtaFixed,
# # "DiffusionModel": DiffusionModel,
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
# "GMM_MLP": GMM_MLP,
"Gaussian_MLP": Gaussian_MLP, 
"Gaussian_VisionMLP": Gaussian_VisionMLP,
# # "GaussianModel": GaussianModel,
# "CriticObs": CriticObs, 
# "CriticObsAct": CriticObsAct,
# # "GMMModel": GMMModel
}



class GaussianModel(tf.keras.Model):

    def __init__(
        self,
        network,
        horizon_steps,
        network_path=None,
        device="cuda:0",
        randn_clip_value=10,
        tanh_output=False,
        **kwargs,
    ):

        print("gaussian.py: GaussianModel.__init__()")



        self.env_name = kwargs.get("env_name", None)

        self.device = device

        print("self.env_name = ", self.env_name)
        


        super().__init__()
        # self.device = device
        # self.network = network
        if not hasattr(self, "network"):
            self.network = network

        # .to(device)

        # if network_path is not None:
        #     print("self.network = ", self.network)
        #     print("GaussianModel: network_path = ", network_path)

            
        #     # checkpoint = tf.train.Checkpoint(model=self)

        #     # print("checkpoint = ", checkpoint)

        #     # checkpoint.restore(network_path).expect_partial()
        #     log.info("Loaded actor from %s", network_path)


        self.network_path = network_path


        if self.network_path is not None:
            print("self.network_path is not None")
            loadpath = network_path

            print("loadpath = ", loadpath)

            if loadpath.endswith(".h5") or loadpath.endswith(".keras"):
                print('loadpath.endswith(".h5") or loadpath.endswith(".keras")')
            else:
                loadpath = network_path.replace('.pt', '.keras')

            # from model.common.mlp_gaussian import Gaussian_MLP
            # from model.common.mlp import MLP, ResidualMLP, TwoLayerPreActivationResNetLinear
            # from model.diffusion.modules import SinusoidalPosEmb
            # from model.common.modules import SpatialEmb, RandomShiftsAug
            # from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish, nn_Identity

            # from model.diffusion.unet import Downsample1d, ResidualBlock1D, Conv1dBlock, Unet1D

            # from util.torch_to_tf import nn_Sequential, nn_Linear, nn_Mish, nn_ReLU,\
            # nn_Conv1d, nn_Identity, einops_layers_torch_Rearrange

            from tensorflow.keras.utils import get_custom_objects

            # cur_dict = {
            #     'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
            #     'Gaussian_MLP': Gaussian_MLP,
            #     # 'VPGDiffusion': VPGDiffusion,
            #     'SinusoidalPosEmb': SinusoidalPosEmb,   
            #     'MLP': MLP,                            # Custom MLP layer
            #     'ResidualMLP': ResidualMLP,            # Custom ResidualMLP layer
            #     'nn_Sequential': nn_Sequential,        # Custom Sequential class
            #     "nn_Identity": nn_Identity,
            #     'nn_Linear': nn_Linear,
            #     'nn_LayerNorm': nn_LayerNorm,
            #     'nn_Dropout': nn_Dropout,
            #     'nn_ReLU': nn_ReLU,
            #     'nn_Mish': nn_Mish,
            #     'SpatialEmb': SpatialEmb,
            #     'RandomShiftsAug': RandomShiftsAug,
            #     "TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
            #     'RandomShiftsAug': RandomShiftsAug,
            #     'Downsample1d': Downsample1d,
            #     'ResidualBlock1D':ResidualBlock1D,
            #     'Conv1dBlock': Conv1dBlock,
            #     'nn_Conv1d': nn_Conv1d,
            #     'Unet1D': Unet1D,
            # }
            # Register your custom class with Keras
            get_custom_objects().update(cur_dict)



            # self = tf.keras.models.load_model(loadpath,  custom_objects=get_custom_objects() )

            final_load_path = loadpath.replace(".keras", "_network.keras")
            print("final_load_path = ", final_load_path)

            self.network = tf.keras.models.load_model( final_load_path ,  custom_objects=get_custom_objects() )


            if OUTPUT_VARIABLES:
                self.output_weights(self.network)

            # self.build_actor(self.network)
            if "ViT" in METHOD_NAME:            
                self.build_actor_vision(self.network)
            else:
                self.build_actor(self.network)



            # print("GaussianModel: self.network = ", self.network )




        # # Log number of parameters in the network
        # num_params = sum(np.prod(var.shape) for var in self.network.trainable_variables)
        # log.info(f"Number of network parameters: {num_params}")

        # for var in self.network.variables:
        #     print(f"Gaussian.network: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")


        self.horizon_steps = horizon_steps

        # Clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Whether to apply tanh to the **sampled** action --- used in SAC
        self.tanh_output = tanh_output







    def get_config(self):
        config = super(GaussianModel, self).get_config()

        # config = {}

        if OUTPUT_FUNCTION_HEADER:
            print("get_config: diffusion.py:GaussianModel.get_config()")


        if OUTPUT_VARIABLES:
            # Debugging each attribute to make sure they are initialized correctly
            print(f"network: {self.network}")
            print(f"device: {self.device}")
            print(f"horizon_steps: {self.horizon_steps}")
            print(f"network: {self.network}")
            print(f"network_path: {self.network_path}")
            print(f"randn_clip_value: {self.randn_clip_value}")
            print(f"tanh_output: {self.tanh_output}")



        from model.common.mlp_gaussian import Gaussian_MLP
        from model.diffusion.unet import Unet1D

        if isinstance( self.network, (Gaussian_MLP, Unet1D, Gaussian_VisionMLP) ):
            network_repr = self.network.get_config()
            if OUTPUT_VARIABLES:
                print("network_repr = ", network_repr)
        else:
            if OUTPUT_VARIABLES:
                print("type(self.network) = ", type(self.network))
            raise RuntimeError("not recognozed type of self.network")



        config.update({
            "device": self.device,
            "horizon_steps": self.horizon_steps,
            "network": network_repr,
            "network_path": self.network_path,
            "randn_clip_value": self.randn_clip_value,
            "tanh_output": self.tanh_output,
        })



        if hasattr(self, "env_name"):
            print("get_config(): self.env_name = ", self.env_name)
            config.update({
            "env_name": self.env_name,
            })
        else:
            print("get_config(): self.env_name = ", None)
        


        

        if OUTPUT_VARIABLES:
            print("GaussianModel.config = ", config)
        
        return config


    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""

        from model.common.mlp_gaussian import Gaussian_MLP

        # # from model.diffusion.diffusion import DiffusionModel
        # from model.common.mlp import MLP, ResidualMLP, TwoLayerPreActivationResNetLinear
        # from model.diffusion.modules import SinusoidalPosEmb
        # from model.common.modules import SpatialEmb, RandomShiftsAug
        # from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, \
        #     nn_Dropout, nn_ReLU, nn_Mish, nn_Identity, nn_Conv1d, nn_ConvTranspose1d

        # from model.diffusion.unet import Unet1D, ResidualBlock1D


        from tensorflow.keras.utils import get_custom_objects

        # cur_dict = {
        #     # 'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
        #     'Gaussian_MLP': Gaussian_MLP,
        #     # 'VPGDiffusion': VPGDiffusion,
        #     'SinusoidalPosEmb': SinusoidalPosEmb,   
        #     'MLP': MLP,                            # Custom MLP layer
        #     'ResidualMLP': ResidualMLP,            # Custom ResidualMLP layer
        #     'nn_Sequential': nn_Sequential,        # Custom Sequential class
        #     "nn_Identity": nn_Identity,
        #     'nn_Linear': nn_Linear,
        #     'nn_LayerNorm': nn_LayerNorm,
        #     'nn_Dropout': nn_Dropout,
        #     'nn_ReLU': nn_ReLU,
        #     'nn_Mish': nn_Mish,
        #     'SpatialEmb': SpatialEmb,
        #     'RandomShiftsAug': RandomShiftsAug,
        #     "TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
        #     "Unet1D": Unet1D,
        #     "ResidualBlock1D": ResidualBlock1D,
        #     "nn_Conv1d": nn_Conv1d,
        #     "nn_ConvTranspose1d": nn_ConvTranspose1d
        #  }
        # # Register your custom class with Keras
        get_custom_objects().update(cur_dict)

        # print('get_custom_objects() = ', get_custom_objects())

        network = config.pop("network")

        if OUTPUT_VARIABLES:
            print("GaussianModel from_config(): network = ", network)

        name = network["name"]
    
        # if OUTPUT_VARIABLES:
        print("network['name'] = ", name)

        if name.startswith("gaussian_mlp"):
            network = Gaussian_MLP.from_config(network)
        elif name.startswith("unet1d"):
            network = Unet1D.from_config(network)
        elif name.startswith("gaussian__vision_mlp"):
            network = Gaussian_VisionMLP.from_config(network)
        else:
            raise RuntimeError("name not recognized")


        # if name in cur_dict:
        #     cur_dict[name].from_config(network)
        # else:
        #     raise RuntimeError("name not recognized")


        result = cls(
            network=network, 
            **config)



        env_name = config.pop("env_name")
        if env_name:
            if OUTPUT_POSITIONS:
                print("Enter env_name")
            result.env_name = env_name
        else:
            result.env_name = None

        return result






    def loss_ori(
        self,
        training,
        true_action,
        cond,
        ent_coef,
    ):
        """no squashing"""

        print("gaussian.py: GaussianModel.loss()")

        B = len(true_action)
        dist = self.forward_train(training, cond, deterministic=False)
        # true_action = tf.reshape(true_action, (B, -1))  # Flatten actions to shape [B, action_dim]
        true_action = torch_tensor_view(true_action, (B, -1))  # Flatten actions to shape [B, action_dim]

        entropy = torch_mean(dist.entropy())
        # log_prob = dist.log_prob(true_action)
        # loss = -torch_mean(log_prob) - entropy * ent_coef

        loss = -dist.log_prob(true_action)  # [B]
        # print("Distribution.Normal log_prob = ", loss)
        loss = torch_mean(loss) - entropy * ent_coef

        return loss, {"entropy": entropy}












    def loss_ori_build(
        self,
        network,
        training,
        true_action,
        cond,
        ent_coef,
    ):
        """no squashing"""

        print("gaussian.py: GaussianModel.loss()")

        B = len(true_action)
        dist = GaussianModel.forward_train_build(self, network, training, cond, deterministic=False)
        # true_action = tf.reshape(true_action, (B, -1))  # Flatten actions to shape [B, action_dim]
        true_action = torch_tensor_view(true_action, (B, -1))  # Flatten actions to shape [B, action_dim]
        log_prob = dist.log_prob(true_action)
        entropy = torch_mean(dist.entropy())
        loss = -torch_mean(log_prob) - entropy * ent_coef
        return loss, {"entropy": entropy}











    def forward_train(
        self,
        training,
        cond,
        deterministic=False,
        network_override=None,
    ):
        """
        Calls the MLP to compute the mean, scale, and logits of the GMM. Returns the torch.Distribution object.
        """

        print("gaussian.py: GaussianModel.forward_train()")

        if network_override is not None:
            print("network_override = ", network_override)
            means, scales = network_override(cond, training=training)
        else:
            print("self.network = ", self.network)
            means, scales = self.network(cond, training=training)
        if deterministic:
            # low-noise for all Gaussian dists
            scales = torch_ones_like(means) * 1e-4

        # dist = tfp.distributions.Normal(loc=means, scale=scales)
        dist = Normal(means, scales)

        return dist




    def forward_train_build(
        self,
        network,
        training,
        cond,
        deterministic=False,
        network_override=None,
    ):
        """
        Calls the MLP to compute the mean, scale, and logits of the GMM. Returns the torch.Distribution object.
        """

        print("gaussian.py: GaussianModel.forward_train()")

        if network_override is not None:
            means, scales = network(cond, training=training)
        else:
            means, scales = network(cond, training=training)
        if deterministic:
            # low-noise for all Gaussian dists
            scales = torch_ones_like(means) * 1e-4

        # dist = tfp.distributions.Normal(loc=means, scale=scales)
        dist = Normal(means, scales)

        return dist










    def call(
        self,
        cond,
        deterministic=False,
        network_override=None,
        reparameterize=False,
        get_logprob=False,
        training = True,
    ):

        print("gaussian.py: GaussianModel.call()")

        # B = len(cond["state"]) if "state" in cond else len(cond["rgb"])
        B = cond["state"].shape[0] if "state" in cond else cond["rgb"].shape[0]

        T = self.horizon_steps
        dist = self.forward_train(
            training,
            cond,
            deterministic=deterministic,
            network_override=network_override,
        )

        if reparameterize:
            sampled_action = dist.rsample()  # reparameterized sample
        else:
            sampled_action = dist.sample()  # standard sample


        # Clipping the sampled action (similar to PyTorch clamp_)
        sampled_action = torch_clamp(sampled_action, dist.loc - self.randn_clip_value * dist.scale, dist.loc + self.randn_clip_value * dist.scale)

        if get_logprob:
            log_prob = dist.log_prob(sampled_action)

            # For SAC/RLPD, squash mean after sampling here instead of right after model output as in PPO
            if self.tanh_output:
                sampled_action = torch_tanh(sampled_action)
                log_prob -= torch_log(1 - tf.square(sampled_action) + 1e-6)

            return torch_tensor_view(sampled_action, [B, T, -1]), torch_sum(log_prob, dim=1, keepdim=False)
        else:
            if self.tanh_output:
                sampled_action = torch_tanh(sampled_action)
            return torch_tensor_view(sampled_action, (B, T, -1))

            






    def build_actor(self, actor, shape1=None, shape2=None):
        # return
    
        print("build_actor: self.env_name = ", self.env_name)

        if shape1 != None and shape2 != None:
            pass
        # Gym - hopper/walker2d/halfcheetah
        elif self.env_name == "hopper-medium-v2":
            # hopper_medium
            # item_actions_copy.shape =  
            shape1 = (128, 4, 3)
            # cond_copy['state'].shape =  
            shape2 = (128, 1, 11)
        elif self.env_name == "kitchen-complete-v0":
            shape1 = (128, 4, 9)
            shape2 = (128, 1, 60)
        elif self.env_name == "kitchen-mixed-v0":
            shape1 = (256, 4, 9)
            shape2 = (256, 1, 60)
        elif self.env_name == "kitchen-partial-v0":
            shape1 = (128, 4, 9)
            shape2 = (128, 1, 60)
        elif self.env_name == "walker2d-medium-v2":
            shape1 = (128, 4, 6)
            shape2 = (128, 1, 17)
        elif self.env_name == "halfcheetah-medium-v2":
            shape1 = (128, 4, 6)
            shape2 = (128, 1, 17)
        # Robomimic - lift/can/square/transport
        elif self.env_name == "lift":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 19)

        elif self.env_name == "can":
            #can 
            # item_actions_copy.shape =  
            shape1 = (256, 4, 7)
            # cond_copy['state'].shape =  
            shape2 = (256, 1, 23)

        elif self.env_name == "square":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 23)

        elif self.env_name == "transport":
            shape1 = (256, 8, 14)
            shape2 = (256, 1, 59)

        # the same name "avoiding-m5" for D3IL with avoid_m1/m2/m3
        elif self.env_name == "avoiding-m5" or self.env_name == "avoid":
            #avoid_m1
            # item_actions_copy.shape =  
            shape1 = (16, 4, 2)
            # cond_copy['state'].shape =  
            shape2 = (16, 1, 4)

        # Furniture-Bench - one_leg/lamp/round_table_low/med
        elif self.env_name == "lamp_low_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 44)
        elif self.env_name == "lamp_med_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 44)
        elif self.env_name == "one_leg_low_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 58)
        elif self.env_name == "one_leg_med_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 58)
        elif self.env_name == "round_table_low_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 44)
        elif self.env_name == "round_table_med_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 44)
        
        else:
            # #one_leg_low
            # # item_actions_copy.shape =  
            # shape1 = (256, 8, 10)
            # # cond_copy['state'].shape =  
            # shape2 = (256, 1, 58)
            raise RuntimeError("The build shape is not implemented for current dataset")


        # param1 = tf.constant(np.random.randn(*shape1).astype(np.float32))
        # param2 = tf.constant(np.random.randn(*shape2).astype(np.float32))


        if OUTPUT_VARIABLES:
            print("type(shape1) = ", type(shape1))
            print("type(shape2) = ", type(shape2))

            print("shape1 = ", shape1)
            print("shape2 = ", shape2)


        param1 = torch_ones(*shape1)
        param2 = torch_ones(*shape2)

        build_dict = {'state': param2}


        
        # _ = self.loss_ori(param1, build_dict)
        all_one_build_result = self.loss_ori_build(actor, training=False, true_action = param1, cond=build_dict, ent_coef = 0)

        print("all_one_build_result = ", all_one_build_result)










    def build_actor_vision(self, actor, shape1=None, shape2=None):
        # return
    
        print("build_actor_vision: self.env_name = ", self.env_name)

        if shape1 != None and shape2 != None:
            pass

        elif self.env_name == "square":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 9)
            shape3 = (256, 1, 3, 96, 96)     
        elif self.env_name == "transport":
            shape1 =  (256, 8, 14)
            shape2 =  (256, 1, 18)
            shape3 =  (256, 1, 6, 96, 96)
        else:
            raise RuntimeError("The build shape is not implemented for current dataset")


        param1 = torch_ones(*shape1)
        param2 = torch_ones(*shape2)
        param3 = torch_ones(*shape3)

        build_dict = {'state': param2}
        build_dict['rgb'] = param3


        all_one_build_result = self.loss_ori_build(actor, training=False, true_action = param1, cond=build_dict, ent_coef = 0)

        print("all_one_build_result = ", all_one_build_result)






    def load_pickle_gaussian_mlp(self, network_path):
        pkl_file_path = network_path.replace('.pt', '_model.pkl')

        print("pkl_file_path = ", pkl_file_path)

        import pickle
        # load pickle file
        with open(pkl_file_path, 'rb') as file:
            params_dict = pickle.load(file)



        # print loaded content

        if OUTPUT_VARIABLES:
            print("params_dict = ", params_dict)


        # # Square
        # 'network.logvar_min'
        # 'network.logvar_max'
        # 'network.mlp_mean.layers.0.weight'
        # 'network.mlp_mean.layers.0.bias'
        # 'network.mlp_mean.layers.1.l1.weight'
        # 'network.mlp_mean.layers.1.l1.bias'
        # 'network.mlp_mean.layers.1.l2.weight'
        # 'network.mlp_mean.layers.1.l2.bias'
        # 'network.mlp_mean.layers.2.weight'
        # 'network.mlp_mean.layers.2.bias'

        self.network.logvar_min = nn_Parameter(
            torch_tensor(params_dict['network.logvar_min']), requires_grad=False
        )

        self.network.logvar_max = nn_Parameter(
            torch_tensor(params_dict['network.logvar_max']), requires_grad=False
        )


        if 'network.mlp_mean.layers.0.weight' in params_dict:
            self.network.mlp_mean.my_layers[0].trainable_weights[0].assign(params_dict['network.mlp_mean.layers.0.weight'].T)  # kernel
        if 'network.mlp_mean.layers.0.bias' in params_dict:
            self.network.mlp_mean.my_layers[0].trainable_weights[1].assign(params_dict['network.mlp_mean.layers.0.bias'])     # bias

        if 'network.mlp_mean.layers.1.l1.weight' in params_dict:
            self.network.mlp_mean.my_layers[1].l1.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.1.l1.weight'].T)  # kernel
        if 'network.mlp_mean.layers.1.l1.bias' in params_dict:
            self.network.mlp_mean.my_layers[1].l1.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.1.l1.bias'])     # bias

        if 'network.mlp_mean.layers.1.l2.weight' in params_dict:
            self.network.mlp_mean.my_layers[1].l2.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.1.l2.weight'].T)  # kernel
        if 'network.mlp_mean.layers.1.l2.bias' in params_dict:
            self.network.mlp_mean.my_layers[1].l2.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.1.l2.bias'])     # bias

        if 'network.mlp_mean.layers.2.weight' in params_dict:
            self.network.mlp_mean.my_layers[2].trainable_weights[0].assign(params_dict['network.mlp_mean.layers.2.weight'].T)  # kernel
        if 'network.mlp_mean.layers.2.bias' in params_dict:
            self.network.mlp_mean.my_layers[2].trainable_weights[1].assign(params_dict['network.mlp_mean.layers.2.bias'])     # bias






    def load_pickle_gaussian_mlp_img(self, network_path):
        pkl_file_path = network_path.replace('.pt', '_model.pkl')

        print("pkl_file_path = ", pkl_file_path)

        import pickle
        # load pickle file
        with open(pkl_file_path, 'rb') as file:
            params_dict = pickle.load(file)



        # print loaded content

        if OUTPUT_VARIABLES:
            print("params_dict = ", params_dict)
        # Square
        # 'network.logvar_min'
        # 'network.logvar_max'
        # 'network.backbone.vit.pos_embed'
        # 'network.backbone.vit.patch_embed.embed.0.weight'
        # 'network.backbone.vit.patch_embed.embed.0.bias'
        # 'network.backbone.vit.patch_embed.embed.3.weight'
        # 'network.backbone.vit.patch_embed.embed.3.bias'
        # 'network.backbone.vit.net.0.layer_norm1.weight'
        # 'network.backbone.vit.net.0.layer_norm1.bias'
        # 'network.backbone.vit.net.0.mha.qkv_proj.weight'
        # 'network.backbone.vit.net.0.mha.qkv_proj.bias'
        # 'network.backbone.vit.net.0.mha.out_proj.weight'
        # 'network.backbone.vit.net.0.mha.out_proj.bias'
        # 'network.backbone.vit.net.0.layer_norm2.weight'
        # 'network.backbone.vit.net.0.layer_norm2.bias'
        # 'network.backbone.vit.net.0.linear1.weight'
        # 'network.backbone.vit.net.0.linear1.bias'
        # 'network.backbone.vit.net.0.linear2.weight'
        # 'network.backbone.vit.net.0.linear2.bias'
        # 'network.backbone.vit.norm.weight'
        # 'network.backbone.vit.norm.bias'
        # 'network.compress.weight'
        # 'network.compress.input_proj.0.weight'
        # 'network.compress.input_proj.0.bias'
        # 'network.compress.input_proj.1.weight'
        # 'network.compress.input_proj.1.bias'
        # 'network.mlp_mean.layers.0.weight'
        # 'network.mlp_mean.layers.0.bias'
        # 'network.mlp_mean.layers.1.l1.weight'
        # 'network.mlp_mean.layers.1.l1.bias'
        # 'network.mlp_mean.layers.1.l2.weight'
        # 'network.mlp_mean.layers.1.l2.bias'
        # 'network.mlp_mean.layers.2.weight'
        # 'network.mlp_mean.layers.2.bias'
        

        # Transport
        # 'network.logvar_min'
        # 'network.logvar_max'
        # 'network.backbone.vit.pos_embed'
        # 'network.backbone.vit.patch_embed.embed.0.weight'
        # 'network.backbone.vit.patch_embed.embed.0.bias'
        # 'network.backbone.vit.patch_embed.embed.3.weight'
        # 'network.backbone.vit.patch_embed.embed.3.bias'
        # 'network.backbone.vit.net.0.layer_norm1.weight'
        # 'network.backbone.vit.net.0.layer_norm1.bias'
        # 'network.backbone.vit.net.0.mha.qkv_proj.weight'
        # 'network.backbone.vit.net.0.mha.qkv_proj.bias'
        # 'network.backbone.vit.net.0.mha.out_proj.weight'
        # 'network.backbone.vit.net.0.mha.out_proj.bias'
        # 'network.backbone.vit.net.0.layer_norm2.weight'
        # 'network.backbone.vit.net.0.layer_norm2.bias'
        # 'network.backbone.vit.net.0.linear1.weight'
        # 'network.backbone.vit.net.0.linear1.bias'
        # 'network.backbone.vit.net.0.linear2.weight'
        # 'network.backbone.vit.net.0.linear2.bias'
        # 'network.backbone.vit.norm.weight'
        # 'network.backbone.vit.norm.bias'
        # 'network.compress1.weight'
        # 'network.compress1.input_proj.0.weight'
        # 'network.compress1.input_proj.0.bias'
        # 'network.compress1.input_proj.1.weight'
        # 'network.compress1.input_proj.1.bias'
        # 'network.compress2.weight'
        # 'network.compress2.input_proj.0.weight'
        # 'network.compress2.input_proj.0.bias'
        # 'network.compress2.input_proj.1.weight'
        # 'network.compress2.input_proj.1.bias'
        # 'network.mlp_mean.layers.0.weight'
        # 'network.mlp_mean.layers.0.bias'
        # 'network.mlp_mean.layers.1.l1.weight'
        # 'network.mlp_mean.layers.1.l1.bias'
        # 'network.mlp_mean.layers.1.l2.weight'
        # 'network.mlp_mean.layers.1.l2.bias'
        # 'network.mlp_mean.layers.2.weight'
        # 'network.mlp_mean.layers.2.bias'
        
        
        self.network.logvar_min = nn_Parameter(
            torch_tensor(params_dict['network.logvar_min']), requires_grad=False
        )

        self.network.logvar_max = nn_Parameter(
            torch_tensor(params_dict['network.logvar_max']), requires_grad=False
        )


        if 'network.backbone.vit.pos_embed' in params_dict:
            self.network.backbone.vit.pos_embed = nn_Parameter( torch_tensor(params_dict['network.backbone.vit.pos_embed']) )
            
        if 'network.backbone.vit.patch_embed.embed.0.weight' in params_dict:
            self.network.backbone.vit.patch_embed.embed[0].trainable_weights[0].assign(params_dict['network.backbone.vit.patch_embed.embed.0.weight'].T)  # kernel
        if 'network.backbone.vit.patch_embed.embed.0.bias' in params_dict:
            self.network.backbone.vit.patch_embed.embed[0].trainable_weights[1].assign(params_dict['network.backbone.vit.patch_embed.embed.0.bias'])  # bias

        if 'network.backbone.vit.patch_embed.embed.3.weight' in params_dict:
            self.network.backbone.vit.patch_embed.embed[3].trainable_weights[0].assign(params_dict['network.backbone.vit.patch_embed.embed.3.weight'].T)  # kernel
        if 'network.backbone.vit.patch_embed.embed.3.bias' in params_dict:
            self.network.backbone.vit.patch_embed.embed[3].trainable_weights[1].assign(params_dict['network.backbone.vit.patch_embed.embed.3.bias'])  # bias




        if 'network.backbone.vit.net.0.layer_norm1.weight' in params_dict:
            self.network.backbone.vit.net[0].layer_norm1.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.layer_norm1.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.layer_norm1.bias' in params_dict:
            self.network.backbone.vit.net[0].layer_norm1.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.layer_norm1.bias'])  # bias

        if 'network.backbone.vit.net.0.mha.qkv_proj.weight' in params_dict:
            self.network.backbone.vit.net[0].mha.qkv_proj.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.mha.qkv_proj.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.mha.qkv_proj.bias' in params_dict:
            self.network.backbone.vit.net[0].mha.qkv_proj.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.mha.qkv_proj.bias'])  # bias


        if 'network.backbone.vit.net.0.mha.out_proj.weight' in params_dict:
            self.network.backbone.vit.net[0].mha.out_proj.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.mha.out_proj.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.mha.out_proj.bias' in params_dict:
            self.network.backbone.vit.net[0].mha.out_proj.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.mha.out_proj.bias'])  # bias

        if 'network.backbone.vit.net.0.layer_norm2.weight' in params_dict:
            self.network.backbone.vit.net[0].layer_norm2.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.layer_norm2.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.layer_norm2.bias' in params_dict:
            self.network.backbone.vit.net[0].layer_norm2.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.layer_norm2.bias'])  # bias


        if 'network.backbone.vit.net.0.linear1.weight' in params_dict:
            self.network.backbone.vit.net[0].linear1.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.linear1.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.linear1.bias' in params_dict:
            self.network.backbone.vit.net[0].linear1.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.linear1.bias'])  # bias


        if 'network.backbone.vit.net.0.linear2.weight' in params_dict:
            self.network.backbone.vit.net[0].linear2.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.linear2.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.linear2.bias' in params_dict:
            self.network.backbone.vit.net[0].linear2.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.linear2.bias'])  # bias


        if 'network.backbone.vit.norm.weight' in params_dict:
            self.network.backbone.vit.norm.trainable_weights[0].assign(params_dict['network.backbone.vit.norm.weight'].T)  # kernel
        if 'network.backbone.vit.norm.bias' in params_dict:
            self.network.backbone.vit.norm.trainable_weights[1].assign(params_dict['network.backbone.vit.norm.bias'])  # bias





        # print("self.network.compress = ", self.network.compress)
        # assert 0 == 1, "network.compress check"
        # 'network.compress.weight'
        if 'network.compress.weight' in params_dict:
            if isinstance(self.network.compress.weight, tf.Variable):       
                self.network.compress.weight = nn_Parameter(
                    torch_tensor(params_dict['network.compress.weight']), requires_grad=False
                )
            else:
                self.network.compress.weight.trainable_weights[0].assign(params_dict['network.compress.weight'])  # kernel


        if 'network.compress.input_proj.0.weight' in params_dict:
            self.network.compress.input_proj[0].trainable_weights[0].assign(params_dict['network.compress.input_proj.0.weight'].T)  # kernel
        if 'network.compress.input_proj.0.bias' in params_dict:
            self.network.compress.input_proj[0].trainable_weights[1].assign(params_dict['network.compress.input_proj.0.bias'])  # bias

        if 'network.compress.input_proj.1.weight' in params_dict:
            self.network.compress.input_proj[1].trainable_weights[0].assign(params_dict['network.compress.input_proj.1.weight'].T)  # kernel
        if 'network.compress.input_proj.1.bias' in params_dict:
            self.network.compress.input_proj[1].trainable_weights[1].assign(params_dict['network.compress.input_proj.1.bias'])  # bias







        if 'network.compress1.weight' in params_dict:
            if isinstance(self.network.compress1.weight, tf.Variable):       
                self.network.compress1.weight = nn_Parameter(
                    torch_tensor(params_dict['network.compress1.weight']), requires_grad=False
                )
            else:
                self.network.compress1.weight.trainable_weights[0].assign(params_dict['network.compress1.weight'])  # kernel


        if 'network.compress1.input_proj.0.weight' in params_dict:
            self.network.compress1.input_proj[0].trainable_weights[0].assign(params_dict['network.compress1.input_proj.0.weight'].T)  # kernel
        if 'network.compress1.input_proj.0.bias' in params_dict:
            self.network.compress1.input_proj[0].trainable_weights[1].assign(params_dict['network.compress1.input_proj.0.bias'])  # bias

        if 'network.compress1.input_proj.1.weight' in params_dict:
            self.network.compress1.input_proj[1].trainable_weights[0].assign(params_dict['network.compress1.input_proj.1.weight'].T)  # kernel
        if 'network.compress1.input_proj.1.bias' in params_dict:
            self.network.compress1.input_proj[1].trainable_weights[1].assign(params_dict['network.compress1.input_proj.1.bias'])  # bias



        if 'network.compress2.weight' in params_dict:
            if isinstance(self.network.compress2.weight, tf.Variable):       
                self.network.compress2.weight = nn_Parameter(
                    torch_tensor(params_dict['network.compress2.weight']), requires_grad=False
                )
            else:
                self.network.compress2.weight.trainable_weights[0].assign(params_dict['network.compress2.weight'])  # kernel

        if 'network.compress2.input_proj.0.weight' in params_dict:
            self.network.compress2.input_proj[0].trainable_weights[0].assign(params_dict['network.compress2.input_proj.0.weight'].T)  # kernel
        if 'network.compress2.input_proj.0.bias' in params_dict:
            self.network.compress2.input_proj[0].trainable_weights[1].assign(params_dict['network.compress2.input_proj.0.bias'])  # bias

        if 'network.compress2.input_proj.1.weight' in params_dict:
            self.network.compress2.input_proj[1].trainable_weights[0].assign(params_dict['network.compress2.input_proj.1.weight'].T)  # kernel
        if 'network.compress2.input_proj.1.bias' in params_dict:
            self.network.compress2.input_proj[1].trainable_weights[1].assign(params_dict['network.compress2.input_proj.1.bias'])  # bias










        if 'network.mlp_mean.layers.0.weight' in params_dict:
            self.network.mlp_mean.my_layers[0].trainable_weights[0].assign(params_dict['network.mlp_mean.layers.0.weight'].T)  # kernel
        if 'network.mlp_mean.layers.0.bias' in params_dict:
            self.network.mlp_mean.my_layers[0].trainable_weights[1].assign(params_dict['network.mlp_mean.layers.0.bias'])     # bias

        if 'network.mlp_mean.layers.1.l1.weight' in params_dict:
            self.network.mlp_mean.my_layers[1].l1.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.1.l1.weight'].T)  # kernel
        if 'network.mlp_mean.layers.1.l1.bias' in params_dict:
            self.network.mlp_mean.my_layers[1].l1.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.1.l1.bias'])     # bias

        if 'network.mlp_mean.layers.1.l2.weight' in params_dict:
            self.network.mlp_mean.my_layers[1].l2.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.1.l2.weight'].T)  # kernel
        if 'network.mlp_mean.layers.1.l2.bias' in params_dict:
            self.network.mlp_mean.my_layers[1].l2.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.1.l2.bias'])     # bias

        if 'network.mlp_mean.layers.2.weight' in params_dict:
            self.network.mlp_mean.my_layers[2].trainable_weights[0].assign(params_dict['network.mlp_mean.layers.2.weight'].T)  # kernel
        if 'network.mlp_mean.layers.2.bias' in params_dict:
            self.network.mlp_mean.my_layers[2].trainable_weights[1].assign(params_dict['network.mlp_mean.layers.2.bias'])     # bias














