"""
GMM policy parameterization.

"""

import tensorflow as tf

import logging

log = logging.getLogger(__name__)

from util.torch_to_tf import Normal, Categorical, Independent, MixtureSameFamily, \
torch_tensor_view, torch_mean, torch_mean, torch_sum, torch_softmax, torch_ones_like, torch_ones,\
nn_Parameter, torch_tensor



from util.config import OUTPUT_VARIABLES, OUTPUT_FUNCTION_HEADER, OUTPUT_POSITIONS






from util.torch_to_tf import nn_TransformerEncoder, nn_TransformerEncoderLayer, nn_TransformerDecoder,\
nn_TransformerDecoderLayer, einops_layers_torch_Rearrange, nn_GroupNorm, nn_ConvTranspose1d, nn_Conv2d, nn_Conv1d, \
nn_MultiheadAttention, nn_LayerNorm, nn_Embedding, nn_ModuleList, nn_Sequential, \
nn_Linear, nn_Dropout, nn_ReLU, nn_GELU, nn_ELU, nn_Mish, nn_Softplus, nn_Identity, nn_Tanh
from model.diffusion.unet import ResidualBlock1D, Unet1D
from model.diffusion.modules import Conv1dBlock, Upsample1d, Downsample1d, SinusoidalPosEmb
from model.diffusion.eta import EtaStateAction, EtaState, EtaAction, EtaFixed
from model.common.vit import VitEncoder, PatchEmbed1, PatchEmbed2, MultiHeadAttention, TransformerLayer, MinVit
from model.common.transformer import GMM_Transformer, Transformer
from model.common.modules import SpatialEmb, RandomShiftsAug
from model.common.mlp import MLP, ResidualMLP, TwoLayerPreActivationResNetLinear
from model.common.mlp_gmm import GMM_MLP
from model.common.critic import CriticObs, CriticObsAct


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
"EtaStateAction": EtaStateAction, 
"EtaState": EtaState, 
"EtaAction": EtaAction, 
"EtaFixed": EtaFixed,
#part3:
"VitEncoder": VitEncoder, 
"PatchEmbed1": PatchEmbed1, 
"PatchEmbed2": PatchEmbed2,
"MultiHeadAttention": MultiHeadAttention, 
"TransformerLayer": TransformerLayer, 
"MinVit": MinVit,
"GMM_Transformer": GMM_Transformer, 
"Transformer": Transformer,
"SpatialEmb": SpatialEmb,
"RandomShiftsAug": RandomShiftsAug,
"MLP": MLP,
"ResidualMLP": ResidualMLP, 
"TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
"GMM_MLP": GMM_MLP,
"CriticObs": CriticObs, 
"CriticObsAct": CriticObsAct,
}








class GMMModel(tf.keras.Model):

    def __init__(
        self,
        network,
        horizon_steps,
        network_path=None,
        device="cuda:0",
        **kwargs,
    ):

        print("gmm.py: GMMModel.__init__()")

        self.env_name = kwargs.get("env_name", None)

        print("self.env_name = ", self.env_name)

        
        super().__init__()
        self.network = network
        self.network_path = network_path
        self.device = device


        if self.network_path is not None:
            print("self.network_path is not None")
            loadpath = network_path

            print("loadpath = ", loadpath)

            if loadpath.endswith(".h5") or loadpath.endswith(".keras"):
                print('loadpath.endswith(".h5") or loadpath.endswith(".keras")')
            else:
                loadpath = network_path.replace('.pt', '.keras')


            from tensorflow.keras.utils import get_custom_objects

            # Register custom class with Keras
            get_custom_objects().update(cur_dict)



            final_load_path = loadpath.replace(".keras", "_network.keras")
            print("final_load_path = ", final_load_path)

            self.network = tf.keras.models.load_model( final_load_path ,  custom_objects=get_custom_objects() )


            if OUTPUT_VARIABLES:
                self.output_weights(self.network)

            self.build_actor(self.network)
            



        log.info(
            f"Number of network parameters: {sum(var.numpy().size for var in self.trainable_variables)}"
        )
        for var in self.network.variables:
            print(f"GMM.network: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")

        self.horizon_steps = horizon_steps







    def get_config(self):
        config = super(GMMModel, self).get_config()


        if OUTPUT_FUNCTION_HEADER:
            print("get_config: gmm.py:GMMModel.get_config()")


        if OUTPUT_VARIABLES:
            # Debugging each attribute to make sure they are initialized correctly
            print(f"network: {self.network}")
            print(f"device: {self.device}")
            print(f"horizon_steps: {self.horizon_steps}")

            print(f"network_path: {self.network_path}")
            print(f"tanh_output: {self.tanh_output}")




        from model.diffusion.unet import Unet1D

        if isinstance( self.network, (GMM_MLP, Unet1D) ):
            network_repr = self.network.get_config()
            if OUTPUT_VARIABLES:
                print("network_repr = ", network_repr)
        else:
            # if OUTPUT_VARIABLES:
            print("type(self.network) = ", type(self.network))
            raise RuntimeError("not recognozed type of self.network")



        config.update({
            "device": self.device,
            "horizon_steps": self.horizon_steps,
            "network": network_repr,
            "network_path": self.network_path,
        })



        if hasattr(self, "env_name"):
            print("get_config(): self.env_name = ", self.env_name)
            config.update({
            "env_name": self.env_name,
            })
        else:
            print("get_config(): self.env_name = ", None)
        


        

        if OUTPUT_VARIABLES:
            print("GMMModel.config = ", config)
        
        return config


    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""


        from tensorflow.keras.utils import get_custom_objects


        # Register custom class with Keras
        get_custom_objects().update(cur_dict)


        network = config.pop("network")

        if OUTPUT_VARIABLES:
            print("GMMModel from_config(): network = ", network)

        name = network["name"]
    
        # if OUTPUT_VARIABLES:
        print("network['name'] = ", name)

        if name.startswith("gmm_mlp"):
            network = GMM_MLP.from_config(network)
        elif name.startswith("unet1d"):
            network = Unet1D.from_config(network)
        else:
            raise RuntimeError("name not recognized")



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
        *args,
        **kwargs,
    ):

        print("gmm.py: GMMModel.loss()")

        B = tf.shape(true_action)[0]

        dist, entropy, _ = self.forward_train(
            training,
            cond,
            deterministic=False,
        )
        true_action = torch_tensor_view(true_action, [B, -1])
        loss = -dist.log_prob(true_action)  # [B]
        loss = torch_mean(loss)
        return loss, {"entropy": entropy}























    def loss_ori_build(
        self,
        network,
        training,
        true_action,
        cond,
        *args,
        **kwargs,
    ):

        print("gmm.py: GMMModel.loss()")

        B = tf.shape(true_action)[0]

        dist, entropy, _ = self.forward_train_build(
            network,
            training,
            cond,
            deterministic=False,
        )
        true_action = torch_tensor_view(true_action, [B, -1])
        loss = -dist.log_prob(true_action)  # [B]
        loss = torch_mean(loss)
        return loss, {"entropy": entropy}

















    def forward_train(
        self,
        training,
        cond,
        deterministic=False,
    ):
        """
        Calls the MLP to compute the mean, scale, and logits of the GMM. Returns the torch.Distribution object.
        """

        print("gmm.py: GMMModel.forward_train()")
        means, scales, logits = self.network(cond, training=training)


        if deterministic:
            # low-noise for all Gaussian dists
            scales = torch_ones_like(means) * 1e-4

        # mixture components - make sure that `batch_shape` for the distribution is equal to (batch_size, num_modes) since MixtureSameFamily expects this shape
        # Each mode has mean vector of dim T*D

        component_distribution = Normal(means, scales)

        component_distribution = Independent(component_distribution, 1)

        component_entropy = component_distribution.entropy()



        approx_entropy = torch_mean(
            torch_sum(torch_softmax(logits, dim=-1) * component_entropy, dim=-1)
        )
        
        std = torch_mean(torch_sum(torch_softmax(logits, dim=-1) * torch_mean(scales, dim=-1), dim=-1))

        # Unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = Categorical(logits=logits)
        
        dist = MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        
        
        return dist, approx_entropy, std
    



















    def forward_train_build(
        self,
        network,
        training,
        cond,
        deterministic=False,
    ):
        """
        Calls the MLP to compute the mean, scale, and logits of the GMM. Returns the torch.Distribution object.
        """

        print("gmm.py: GMMModel.forward_train()")

        means, scales, logits = network(cond, training=training)
        if deterministic:
            # low-noise for all Gaussian dists
            scales = torch_ones_like(means) * 1e-4

        # mixture components - make sure that `batch_shape` for the distribution is equal to (batch_size, num_modes) since MixtureSameFamily expects this shape
        # Each mode has mean vector of dim T*D

        # component_distribution = tfp.distributions.Normal(loc=means, scale=scales)
        component_distribution = Normal(means, scales)

        component_distribution = Independent(component_distribution, 1)

        component_entropy = component_distribution.entropy()



        approx_entropy = torch_mean(
            torch_sum(torch_softmax(logits, dim=-1) * component_entropy, dim=-1)
        )
        
        std = torch_mean(torch_sum(torch_softmax(logits, dim=-1) * torch_mean(scales, dim=-1), dim=-1))

        # Unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = Categorical(logits=logits)
        
        dist = MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        
        
        return dist, approx_entropy, std
    







    def call(self, cond, deterministic=False, training=True):

        print("gmm.py: GMMModel.call()")

        B = tf.shape(cond["state"])[0] if "state" in cond else tf.shape(cond["rgb"])[0]

        T = self.horizon_steps
        dist, _, _ = self.forward_train(
            training,
            cond,
            deterministic=deterministic,
        )


        sampled_action = dist.sample()


        sampled_action = torch_tensor_view(sampled_action, [B, T, -1])
        return sampled_action
    













    def load_pickle_gmm_mlp(self, network_path):
        pkl_file_path = network_path.replace('.pt', '_model.pkl')

        print("pkl_file_path = ", pkl_file_path)

        import pickle
        # load pickle file
        with open(pkl_file_path, 'rb') as file:
            params_dict = pickle.load(file)


        if OUTPUT_VARIABLES:
            print("params_dict = ", params_dict)



        self.logvar_min = nn_Parameter(
            torch_tensor(params_dict['network.logvar_min']), requires_grad=False
        )

        self.logvar_max = nn_Parameter(
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



        if 'network.mlp_weights.layers.0.weight' in params_dict:
            self.network.mlp_weights.my_layers[0].trainable_weights[0].assign(params_dict['network.mlp_weights.layers.0.weight'].T)  # kernel
        if 'network.mlp_weights.layers.0.bias' in params_dict:
            self.network.mlp_weights.my_layers[0].trainable_weights[1].assign(params_dict['network.mlp_weights.layers.0.bias'])     # bias

        if 'network.mlp_weights.layers.1.l1.weight' in params_dict:
            self.network.mlp_weights.my_layers[1].l1.trainable_weights[0].assign(params_dict['network.mlp_weights.layers.1.l1.weight'].T)  # kernel
        if 'network.mlp_weights.layers.1.l1.bias' in params_dict:
            self.network.mlp_weights.my_layers[1].l1.trainable_weights[1].assign(params_dict['network.mlp_weights.layers.1.l1.bias'])     # bias

        if 'network.mlp_weights.layers.1.l2.weight' in params_dict:
            self.network.mlp_weights.my_layers[1].l2.trainable_weights[0].assign(params_dict['network.mlp_weights.layers.1.l2.weight'].T)  # kernel
        if 'network.mlp_weights.layers.1.l2.bias' in params_dict:
            self.network.mlp_weights.my_layers[1].l2.trainable_weights[1].assign(params_dict['network.mlp_weights.layers.1.l2.bias'])     # bias

        if 'network.mlp_weights.layers.2.weight' in params_dict:
            self.network.mlp_weights.my_layers[2].trainable_weights[0].assign(params_dict['network.mlp_weights.layers.2.weight'].T)  # kernel
        if 'network.mlp_weights.layers.2.bias' in params_dict:
            self.network.mlp_weights.my_layers[2].trainable_weights[1].assign(params_dict['network.mlp_weights.layers.2.bias'])     # bias










    def build_actor(self, actor, shape1=None, shape2=None):
    
        print("build_actor: self.env_name = ", self.env_name)

        if shape1 != None and shape2 != None:
            pass
        # Gym - hopper/walker2d/halfcheetah
        elif self.env_name == "hopper-medium-v2":
            shape1 = (128, 4, 3)
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
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 23)

        elif self.env_name == "square":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 23)

        elif self.env_name == "transport":
            shape1 = (256, 8, 14)
            shape2 = (256, 1, 59)

        # the same name "avoiding-m5" for D3IL with avoid_m1/m2/m3
        elif self.env_name == "avoiding-m5" or self.env_name == "avoid":
            shape1 = (16, 4, 2)
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
            raise RuntimeError("The build shape is not implemented for current dataset")


        if OUTPUT_VARIABLES:
            print("type(shape1) = ", type(shape1))
            print("type(shape2) = ", type(shape2))

            print("shape1 = ", shape1)
            print("shape2 = ", shape2)


        param1 = torch_ones(*shape1)
        param2 = torch_ones(*shape2)

        build_dict = {'state': param2}


        all_one_build_result = self.loss_ori_build(actor, training=False, true_action = param1, cond=build_dict)

        print("all_one_build_result = ", all_one_build_result)









