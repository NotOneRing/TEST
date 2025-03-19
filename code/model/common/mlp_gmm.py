"""
MLP models for GMM policy.

"""

import tensorflow as tf

import numpy as np

from model.common.mlp import MLP, ResidualMLP

from util.config import OUTPUT_VARIABLES, OUTPUT_FUNCTION_HEADER, OUTPUT_POSITIONS


from util.torch_to_tf import nn_Parameter, torch_log, torch_tensor, torch_tanh, torch_tensor_view, torch_exp, torch_clamp, torch_tensor_repeat, torch_ones_like

from util.torch_to_tf import nn_TransformerEncoder, nn_TransformerEncoderLayer, nn_TransformerDecoder,\
nn_TransformerDecoderLayer, einops_layers_torch_Rearrange, nn_GroupNorm, nn_ConvTranspose1d, nn_Conv2d, nn_Conv1d, \
nn_MultiheadAttention, nn_LayerNorm, nn_Embedding, nn_ModuleList, nn_Sequential, \
nn_Linear, nn_Dropout, nn_ReLU, nn_GELU, nn_ELU, nn_Mish, nn_Softplus, nn_Identity, nn_Tanh
from model.diffusion.unet import ResidualBlock1D, Unet1D
from model.diffusion.modules import Conv1dBlock, Upsample1d, Downsample1d, SinusoidalPosEmb
from model.common.vit import VitEncoder, PatchEmbed1, PatchEmbed2, MultiHeadAttention, TransformerLayer, MinVit
from model.common.transformer import GMM_Transformer, Transformer
from model.common.modules import SpatialEmb, RandomShiftsAug
from model.common.mlp import MLP, ResidualMLP, TwoLayerPreActivationResNetLinear



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
"GMM_Transformer": GMM_Transformer, 
"Transformer": Transformer,
"SpatialEmb": SpatialEmb,
"RandomShiftsAug": RandomShiftsAug,
"MLP": MLP,
"ResidualMLP": ResidualMLP, 
"TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
}

class GMM_MLP(tf.keras.Model):

    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim=None,
        mlp_dims=[256, 256, 256],
        num_modes=5,
        activation_type="Mish",
        residual_style=False,
        use_layernorm=False,
        fixed_std=None,
        learn_fixed_std=False,
        std_min=0.01,
        std_max=1,
        mlp_mean = None,
        mlp_weights = None,
        mlp_logvar = None,
        **kwargs
    ):

        print("mlp_gmm.py: GMM_MLP.__init__()")

        super().__init__()
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps


        self.cond_dim = cond_dim
        self.mlp_dims = list(mlp_dims)
        self.num_modes = num_modes
        self.activation_type = activation_type
        self.residual_style = residual_style
        self.use_layernorm = use_layernorm
        self.std_min = std_min
        self.std_max = std_max


        input_dim = cond_dim
        output_dim = action_dim * horizon_steps * num_modes
        self.num_modes = num_modes

        if residual_style:
            model = ResidualMLP
        else:
            model = MLP

        if mlp_mean:
            self.mlp_mean = mlp_mean
        else:
            self.mlp_mean = model(
                [input_dim] + mlp_dims + [output_dim],
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )

        if fixed_std is None:
            if mlp_logvar:
                self.mlp_logvar = mlp_logvar
            else:            
                self.mlp_logvar = model(
                    [input_dim] + mlp_dims + [output_dim],
                    activation_type=activation_type,
                    out_activation_type="Identity",
                    use_layernorm=use_layernorm,
                )
        elif (
            learn_fixed_std
        ):  # initialize to fixed_std, separate for each action and mode            
            # Learnable fixed_std
            self.logvar = nn_Parameter(
                torch_log(
                    torch_tensor(
                        np.array([fixed_std**2 for _ in range(action_dim * num_modes)])
                    )
                ),
                requires_grad=True,
            )


        self.logvar_min = nn_Parameter(
            torch_log(torch_tensor( np.array( [std_min**2] ) )), requires_grad=False
        )
        self.logvar_max = nn_Parameter(
            torch_log(torch_tensor( np.array( [std_max**2] ) )), requires_grad=False
        )


        self.use_fixed_std = fixed_std is not None
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std

        # mode weights
        if mlp_weights:
            self.mlp_weights = mlp_weights
        else:
            self.mlp_weights = model(
                [input_dim] + mlp_dims + [num_modes],
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )



    def get_config(self):

        if OUTPUT_FUNCTION_HEADER:
            print("GMM_MLP: get_config()")

        config = super(GMM_MLP, self).get_config()



        # print every property with its type and value
        if OUTPUT_VARIABLES:
            print("Checking GMM_MLP Config elements:")
            print(f"action_dim: {self.action_dim}, type: {type(self.action_dim)}")
            print(f"horizon_steps: {self.horizon_steps}, type: {type(self.horizon_steps)}")
            print(f"cond_dim: {self.cond_dim}, type: {type(self.cond_dim)}")
            print(f"mlp_dims: {self.mlp_dims}, type: {type(self.mlp_dims)}")

            print(f"num_modes: {self.num_modes}, type: {type(self.num_modes)}")


            print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")

            print(f"residual_style: {self.residual_style}, type: {type(self.residual_style)}")
            print(f"use_layernorm: {self.use_layernorm}, type: {type(self.use_layernorm)}")
            
            print(f"fixed_std: {self.fixed_std}, type: {type(self.fixed_std)}")
            print(f"learn_fixed_std: {self.learn_fixed_std}, type: {type(self.learn_fixed_std)}")
            print(f"std_min: {self.std_min}, type: {type(self.std_min)}")
            print(f"std_max: {self.std_max}, type: {type(self.std_max)}")
        
        config.update({
            "action_dim": self.action_dim,
            "horizon_steps": self.horizon_steps,
            "cond_dim": self.cond_dim,
            "mlp_dims": self.mlp_dims,
            "num_modes": self.num_modes,
            "activation_type": self.activation_type,
            
            "residual_style": self.residual_style,
            "use_layernorm": self.use_layernorm,

            "fixed_std": self.fixed_std,
            "learn_fixed_std": self.learn_fixed_std,
            "std_min": self.std_min,
            "std_max": self.std_max
        })


        if self.fixed_std is None:
            config.update({
                "mlp_logvar": tf.keras.layers.serialize(self.mlp_logvar),
            })
            

        config.update({
            "mlp_mean": tf.keras.layers.serialize(self.mlp_mean),
            "mlp_weights": tf.keras.layers.serialize(self.mlp_weights),
        })


        return config



    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("GMM_MLP: from_config()")
            
        from tensorflow.keras.utils import get_custom_objects

        get_custom_objects().update(cur_dict)

        fixed_std = config.pop("fixed_std")

        mlp_weights = tf.keras.layers.deserialize(config.pop("mlp_weights"),  custom_objects=get_custom_objects() )

        if fixed_std is None:
            mlp_logvar = tf.keras.layers.deserialize(config.pop("mlp_logvar"),  custom_objects=get_custom_objects() )
        else:
            mlp_logvar = None

        mlp_mean = tf.keras.layers.deserialize(config.pop("mlp_mean") ,  custom_objects=get_custom_objects() )

        result = cls(mlp_weights = mlp_weights, mlp_logvar = mlp_logvar, mlp_mean = mlp_mean, fixed_std = fixed_std, **config)

        return result






    def call(self, cond):

        print("mlp_gmm.py: GMM_MLP.call()")

        B = cond["state"].shape[0]

        # flatten history
        state = torch_tensor_view(cond["state"], [B, -1])

        # mlp
        out_mean = self.mlp_mean(state)


        out_mean = torch_tanh(out_mean)


        out_mean = torch_tensor_view(
            out_mean, [B, self.num_modes, self.horizon_steps * self.action_dim]
        ) # tanh squashing in [-1, 1]


        if self.learn_fixed_std:
            out_logvar = torch_clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch_exp(0.5 * out_logvar)
            out_scale = torch_tensor_view(
                out_scale, [1, self.num_modes, self.action_dim]
            )
            out_scale = torch_tensor_repeat(out_scale, [B, 1, self.horizon_steps])

        elif self.use_fixed_std:
            out_scale = torch_ones_like(out_mean) * self.fixed_std


        else:

            out_logvar = self.mlp_logvar(state)

            out_logvar = torch_tensor_view( out_logvar,
                B, self.num_modes, self.horizon_steps * self.action_dim
            )

            out_logvar = torch_clamp(out_logvar, self.logvar_min, self.logvar_max)

            out_scale = torch_exp(0.5 * out_logvar)


        out_weights = self.mlp_weights(state)

        out_weights = torch_tensor_view(out_weights, [B, self.num_modes])


        return out_mean, out_scale, out_weights











