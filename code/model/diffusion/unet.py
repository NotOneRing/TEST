"""
UNet implementation. Minorly modified from Diffusion Policy: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py

Set `smaller_encoder` to False for using larger observation encoder in ResidualBlock1D

"""

import tensorflow as tf

import einops

from einops import rearrange
from einops.layers.tensorflow import Rearrange


import logging

log = logging.getLogger(__name__)


from model.diffusion.modules import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)
from model.common.mlp import ResidualMLP


from util.torch_to_tf import nn_Sequential, nn_Linear, nn_Mish, nn_ReLU,\
nn_Conv1d, nn_Identity, torch_tensor_expand, torch_cat, torch_reshape, \
einops_layers_torch_Rearrange


from util.config import OUTPUT_FUNCTION_HEADER, OUTPUT_VARIABLES



from util.torch_to_tf import nn_TransformerEncoder, nn_TransformerEncoderLayer, nn_TransformerDecoder,\
nn_TransformerDecoderLayer, einops_layers_torch_Rearrange, nn_GroupNorm, nn_ConvTranspose1d, nn_Conv2d, nn_Conv1d, \
nn_MultiheadAttention, nn_LayerNorm, nn_Embedding, nn_ModuleList, nn_Sequential, \
nn_Linear, nn_Dropout, nn_ReLU, nn_GELU, nn_ELU, nn_Mish, nn_Softplus, nn_Identity, nn_Tanh
from model.diffusion.modules import Conv1dBlock, Upsample1d, Downsample1d, SinusoidalPosEmb
from model.diffusion.eta import EtaStateAction, EtaState, EtaAction, EtaFixed
from model.common.vit import VitEncoder, PatchEmbed1, PatchEmbed2, MultiHeadAttention, TransformerLayer, MinVit
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
"SpatialEmb": SpatialEmb,
"RandomShiftsAug": RandomShiftsAug,
"MLP": MLP,
"ResidualMLP": ResidualMLP, 
"TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
}







class ResidualBlock1D(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=5,
        n_groups=None,
        cond_predict_scale=False,
        larger_encoder=False,
        activation_type="Mish",
        groupnorm_eps=1e-5,
        blocks = None,
        cond_encoder = None,
        residual_conv=None,
        **kwargs
    ):

        print("unet.py: ResidualBlock1D.__init__()")

        super().__init__()

        self.in_channels = in_channels
        self.initial_out_channels = out_channels
        self.cond_dim = cond_dim
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.cond_predict_scale = cond_predict_scale
        self.larger_encoder = larger_encoder
        self.activation_type = activation_type
        self.groupnorm_eps = groupnorm_eps



        self.out_channels = out_channels



        if blocks == None:
            # Convolutional Blocks
            self.blocks = nn_Sequential([
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups, activation_type, groupnorm_eps),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups, activation_type, groupnorm_eps),
            ])
        else:
            self.blocks = blocks


        # Activation Function
        if activation_type == "Mish":
            act = nn_Mish()
        elif activation_type == "ReLU":
            act = nn_ReLU()
        else:
            raise ValueError("Unknown activation type for ResidualBlock1D")



        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2

        self.out_channels = out_channels






        if cond_encoder == None:
            #input is of dimension cond_dim
            if larger_encoder:
                self.cond_encoder = nn_Sequential([
                    nn_Linear(cond_dim, cond_channels),
                    act,
                    nn_Linear(cond_channels, cond_channels),
                    act,
                    nn_Linear(cond_channels, cond_channels),
                    einops_layers_torch_Rearrange("batch t -> batch t 1", name = "ResidualBlock1D_Rearrange1"),
                ])
            else:
                self.cond_encoder = nn_Sequential([
                    act,
                    nn_Linear(cond_dim, cond_channels),
                    einops_layers_torch_Rearrange("batch t -> batch t 1", name = "ResidualBlock1D_Rearrange2"),
                ])
        else:
            self.cond_encoder = cond_encoder


        if residual_conv == None:
            # Residual Connection
            self.residual_conv = (
                nn_Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels
                else nn_Identity()
            )
        else:
            self.residual_conv = residual_conv






    def call(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon_steps ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon_steps ]
        """

        print("unet.py: ResidualBlock1D.forward()")

        print("x = ", x)
        print("self.blocks = ", self.blocks)
        print("self.blocks[0] = ", self.blocks[0])
        
        out = self.blocks[0](x)
        print("out = ", out)

        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = torch_reshape(embed, [embed.shape[0], 2, self.out_channels, 1])
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)

        print("self.residual_conv = ", self.residual_conv)
        print("x.shape = ", x.shape)
        residual_result = self.residual_conv(x)

        return out + residual_result




    def get_config(self):

        if OUTPUT_FUNCTION_HEADER:
            print("ResidualBlock1D: get_config()")

        config = super(ResidualBlock1D, self).get_config()

        if OUTPUT_VARIABLES:
            print("Checking ResidualBlock1D Config elements:")
            print(f"in_channels: {self.in_channels}, type: {type(self.in_channels)}")
            print(f"out_channels: {self.initial_out_channels}, type: {type(self.initial_out_channels)}")
            print(f"cond_dim: {self.cond_dim}, type: {type(self.cond_dim)}")
            print(f"kernel_size: {self.kernel_size}, type: {type(self.kernel_size)}")
            
            print(f"n_groups: {self.n_groups}, type: {type(self.n_groups)}")

            print(f"cond_predict_scale: {self.cond_predict_scale}, type: {type(self.cond_predict_scale)}")
            print(f"larger_encoder: {self.larger_encoder}, type: {type(self.larger_encoder)}")
            print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")

            print(f"groupnorm_eps: {self.groupnorm_eps}, type: {type(self.groupnorm_eps)}")



        config.update({
            "in_channels" : self.in_channels,
            "out_channels" : self.initial_out_channels,
            "cond_dim" : self.cond_dim,
            "kernel_size" : self.kernel_size,
            "n_groups" : self.n_groups,
            "cond_predict_scale" : self.cond_predict_scale,
            "larger_encoder" : self.larger_encoder,
            "activation_type" : self.activation_type,
            "groupnorm_eps" : self.groupnorm_eps
        })


        config.update({
            "blocks": tf.keras.layers.serialize(self.blocks),
            "cond_encoder": tf.keras.layers.serialize(self.cond_encoder),
            "residual_conv": tf.keras.layers.serialize(self.residual_conv),
        })




        return config






    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("ResidualBlock1D: from_config()")

        from tensorflow.keras.utils import get_custom_objects

        cur_dict["ResidualBlock1D"] = ResidualBlock1D
        cur_dict["Unet1D"] = Unet1D


        # Register custom class with Keras
        get_custom_objects().update(cur_dict)

        blocks = tf.keras.layers.deserialize(config.pop("blocks") ,  custom_objects=get_custom_objects() )

        cond_encoder = tf.keras.layers.deserialize(config.pop("cond_encoder"),  custom_objects=get_custom_objects() )

        residual_conv = tf.keras.layers.deserialize(config.pop("residual_conv") ,  custom_objects=get_custom_objects() )


        result = cls(blocks = blocks, cond_encoder = cond_encoder, residual_conv = residual_conv, **config)
        return result








class Unet1D(tf.keras.Model):
        
    def __init__(
        self,
        action_dim,
        cond_dim=None,
        diffusion_step_embed_dim=32,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        smaller_encoder=False,
        cond_mlp_dims=None,
        kernel_size=5,
        n_groups=None,
        activation_type="Mish",
        cond_predict_scale=False,
        groupnorm_eps=1e-5,
        **kwargs
    ):

        dim_mults = list(dim_mults)

        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.dim = dim
        self.dim_mults = dim_mults
        self.smaller_encoder = smaller_encoder
        self.cond_mlp_dims = cond_mlp_dims
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.activation_type = activation_type
        self.cond_predict_scale = cond_predict_scale
        self.groupnorm_eps = groupnorm_eps


        print("unet.py: Unet1D.__init__()")

        super().__init__()
        dims = [action_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        log.info(f"Channel dimensions: {in_out}")

        dsed = diffusion_step_embed_dim
        



        dsed = diffusion_step_embed_dim
        self.time_mlp = nn_Sequential(
            [
            SinusoidalPosEmb(dsed),
            nn_Linear(dsed, dsed * 4),
            nn_Mish(),
            nn_Linear(dsed * 4, dsed),
            ]
        )





        if cond_mlp_dims is not None:
            self.cond_mlp = ResidualMLP(
                dim_list=[cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            cond_block_dim = dsed + cond_mlp_dims[-1]
        else:
            #added
            self.cond_mlp = None

            cond_block_dim = dsed + cond_dim




        use_large_encoder_in_block = cond_mlp_dims is None and not smaller_encoder

        mid_dim = dims[-1]



        # Mid Modules
        self.mid_modules = nn_Sequential([
            ResidualBlock1D(
                mid_dim,
                mid_dim,
                cond_dim=cond_block_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
                larger_encoder=use_large_encoder_in_block,
                activation_type=activation_type,
                groupnorm_eps=groupnorm_eps,
            ),
            ResidualBlock1D(
                mid_dim,
                mid_dim,
                cond_dim=cond_block_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
                larger_encoder=use_large_encoder_in_block,
                activation_type=activation_type,
                groupnorm_eps=groupnorm_eps,
            ),
        ])



       # Down Modules
        self.down_modules = []
        for ind, (dim_in, dim_out) in enumerate(in_out):

            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn_Sequential(
                [
                ResidualBlock1D(
                    dim_in,
                    dim_out,
                    cond_dim=cond_block_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    larger_encoder=use_large_encoder_in_block,
                    activation_type=activation_type,
                    groupnorm_eps=groupnorm_eps,
                ),
                ResidualBlock1D(
                    dim_out,
                    dim_out,
                    cond_dim=cond_block_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    larger_encoder=use_large_encoder_in_block,
                    activation_type=activation_type,
                    groupnorm_eps=groupnorm_eps,
                ),
                Downsample1d(dim_out) if not is_last else nn_Identity(),
            ]
                )
            )

        self.down_modules = nn_Sequential(self.down_modules)



       # Up Modules
        self.up_modules = []
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn_Sequential(
                [
                ResidualBlock1D(
                    dim_out * 2,
                    dim_in,
                    cond_dim=cond_block_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    larger_encoder=use_large_encoder_in_block,
                    activation_type=activation_type,
                    groupnorm_eps=groupnorm_eps,
                ),
                ResidualBlock1D(
                    dim_in,
                    dim_in,
                    cond_dim=cond_block_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    larger_encoder=use_large_encoder_in_block,
                    activation_type=activation_type,
                    groupnorm_eps=groupnorm_eps,
                ),
                Upsample1d(dim_in) if not is_last else nn_Identity(),
            ]
                )
            )

        self.up_modules = nn_Sequential(self.up_modules)


        # Final Conv
        self.final_conv = nn_Sequential([
            Conv1dBlock(
                dim,
                dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                activation_type=activation_type,
                eps=groupnorm_eps,
            ),
            nn_Conv1d(dim, action_dim, 1)
        ])



    def call(
        self,
        # x,
        # time,
        # cond,
        inputs,
        **kwargs,
    ):
        """
        x: (B, Ta, act_dim)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, obs_dim)
        """

        print("unet.py: Unet1D.forward()")

        x, time, cond_state = inputs

        B = x.shape[0]

        # move chunk dim to the end
        x = einops.rearrange(x, "b h t -> b t h")

        # flatten history
        state = tf.reshape( cond_state, [B, -1] )

        # obs encoder
        if hasattr(self, "cond_mlp") and self.cond_mlp:
            state = self.cond_mlp(state)



        # 1. time
        if not tf.is_tensor(time):
            time = tf.convert_to_tensor([time], dtype=tf.int64)
        elif tf.is_tensor(time) and len(time.shape) == 0:
            time = time[None]


        # Broadcast to batch dimension
        time = torch_tensor_expand(time, x.shape[0])

        global_feature = self.time_mlp(time)
        global_feature = torch_cat([global_feature, state], dim=-1)



        # encode local features
        h_local = []
        h = []

        for idx in range(len(self.down_modules)):
            cur_down_modules = self.down_modules[idx]
            
            print("cur_down_modules = ", cur_down_modules)
            print("type(cur_down_modules) = ", type(cur_down_modules))

            resnet, resnet2, downsample = cur_down_modules.layers[0], cur_down_modules.layers[1], cur_down_modules.layers[2]
            

            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx in range(len(self.up_modules)):
            cur_up_modules = self.up_modules[idx]
            resnet, resnet2, upsample = cur_up_modules.layers[0], cur_up_modules.layers[1], cur_up_modules.layers[2]

            x = torch_cat([x, h.pop()], dim=1)  # Concatenate along channel dimension

            x = resnet(x, global_feature)

            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x



    def get_config(self):

        if OUTPUT_FUNCTION_HEADER:
            print("Unet1D: get_config()")

        config = super(Unet1D, self).get_config()



        # print every property with its type and value
        if OUTPUT_VARIABLES:
            print("Checking Unet1D Config elements:")
            print(f"action_dim: {self.action_dim}, type: {type(self.action_dim)}")
            print(f"cond_dim: {self.cond_dim}, type: {type(self.cond_dim)}")
            print(f"diffusion_step_embed_dim: {self.diffusion_step_embed_dim}, type: {type(self.diffusion_step_embed_dim)}")
            print(f"dim: {self.dim}, type: {type(self.dim)}")
            print(f"dim_mults: {self.dim_mults}, type: {type(self.dim_mults)}")
            print(f"smaller_encoder: {self.smaller_encoder}, type: {type(self.smaller_encoder)}")
            print(f"cond_mlp_dims: {self.cond_mlp_dims}, type: {type(self.cond_mlp_dims)}")
            print(f"kernel_size: {self.kernel_size}, type: {type(self.kernel_size)}")
            print(f"n_groups: {self.n_groups}, type: {type(self.n_groups)}")
            print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")
            print(f"cond_predict_scale: {self.cond_predict_scale}, type: {type(self.cond_predict_scale)}")
            print(f"groupnorm_eps: {self.groupnorm_eps}, type: {type(self.groupnorm_eps)}")



        config.update({
            "action_dim" : self.action_dim,
            "cond_dim" : self.cond_dim,
            "diffusion_step_embed_dim" : self.diffusion_step_embed_dim,
            "dim" : self.dim,
            "dim_mults" : self.dim_mults,
            "smaller_encoder" : self.smaller_encoder,
            "cond_mlp_dims" : self.cond_mlp_dims,
            "kernel_size" : self.kernel_size,
            "n_groups" : self.n_groups,
            "activation_type" : self.activation_type,
            "cond_predict_scale" : self.cond_predict_scale,
            "groupnorm_eps" : self.groupnorm_eps
        })




        config.update({
            "time_mlp": tf.keras.layers.serialize(self.time_mlp),
            "mid_modules": tf.keras.layers.serialize(self.mid_modules),
            "cond_mlp": tf.keras.layers.serialize(self.cond_mlp),
            "down_modules": tf.keras.layers.serialize(self.down_modules),
            "up_modules": tf.keras.layers.serialize(self.up_modules),
            "final_conv": tf.keras.layers.serialize(self.final_conv),
                
        })


        return config
    






    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("Unet1D: from_config()")

        from tensorflow.keras.utils import get_custom_objects


        cur_dict["ResidualBlock1D"] = ResidualBlock1D
        cur_dict["Unet1D"] = Unet1D

        # Register custom class with Keras
        get_custom_objects().update(cur_dict)


        time_mlp = tf.keras.layers.deserialize(config.pop("time_mlp") ,  custom_objects=get_custom_objects() )


        config_cond_mlp = config.pop("cond_mlp")
        if config_cond_mlp:
            cond_mlp = tf.keras.layers.deserialize(config_cond_mlp,  custom_objects=get_custom_objects() )
        else:
            cond_mlp = None


        mid_modules = tf.keras.layers.deserialize(config.pop("mid_modules") ,  custom_objects=get_custom_objects() )


        down_modules = tf.keras.layers.deserialize(config.pop("down_modules") ,  custom_objects=get_custom_objects() )


        up_modules = tf.keras.layers.deserialize(config.pop("up_modules") ,  custom_objects=get_custom_objects() )


        final_conv = tf.keras.layers.deserialize(config.pop("final_conv") ,  custom_objects=get_custom_objects() )



        result = cls(time_mlp = time_mlp, cond_mlp = cond_mlp, mid_modules = mid_modules, down_modules = down_modules, up_modules = up_modules, final_conv = final_conv, **config)
        return result




















