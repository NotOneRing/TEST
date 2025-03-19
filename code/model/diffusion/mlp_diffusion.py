import tensorflow as tf
import numpy as np
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug


log = tf.get_logger()

from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish

from util.torch_to_tf import torch_tensor_float, torch_cat, torch_flatten, torch_tensor_view, torch_reshape
from tensorflow.keras.saving import register_keras_serializable


from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, OUTPUT_POSITIONS, OUTPUT_FUNCTION_HEADER









from util.torch_to_tf import nn_TransformerEncoder, nn_TransformerEncoderLayer, nn_TransformerDecoder,\
nn_TransformerDecoderLayer, einops_layers_torch_Rearrange, nn_GroupNorm, nn_ConvTranspose1d, nn_Conv2d, nn_Conv1d, \
nn_MultiheadAttention, nn_LayerNorm, nn_Embedding, nn_ModuleList, nn_Sequential, \
nn_Linear, nn_Dropout, nn_ReLU, nn_GELU, nn_ELU, nn_Mish, nn_Softplus, nn_Identity, nn_Tanh
from model.diffusion.unet import ResidualBlock1D, Unet1D
from model.diffusion.modules import Conv1dBlock, Upsample1d, Downsample1d, SinusoidalPosEmb
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
"SpatialEmb": SpatialEmb,
"RandomShiftsAug": RandomShiftsAug,
"MLP": MLP,
"ResidualMLP": ResidualMLP, 
"TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
}










@register_keras_serializable(package="Custom")
class VisionDiffusionMLP(tf.keras.Model):
    """With ViT backbone"""

    def __init__(
        self,
        backbone,
        action_dim,
        horizon_steps,
        cond_dim,
        img_cond_steps=1,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=0,
        visual_feature_dim=128,
        dropout=0,
        num_img=1,
        augment=False,

        compress = None,
        compress1 = None,
        compress2 = None,
        time_embedding = None,
        mlp_mean = None,
        **kwargs
    ):


        if OUTPUT_FUNCTION_HEADER:
            print("mlp_diffusion.py: VisionDiffusionMLP.__init__()")

        super(VisionDiffusionMLP, self).__init__()

        # vision
        self.backbone = backbone
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.cond_dim = cond_dim


        if not isinstance(mlp_dims, list):
            self.mlp_dims = list(mlp_dims)
        else:
            self.mlp_dims = mlp_dims



        self.activation_type = activation_type
        self.out_activation_type = out_activation_type
        self.use_layernorm = use_layernorm
        self.residual_style = residual_style
        self.spatial_emb = spatial_emb
        self.visual_feature_dim = visual_feature_dim      
        self.dropout = dropout
        self.num_img = num_img
        self.augment = augment


        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        self.num_img = num_img

        self.img_cond_steps = img_cond_steps
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
            if num_img > 1:
                if compress1:
                    self.compress1 = compress1
                else:
                    self.compress1 = SpatialEmb(
                        num_patch=self.backbone.num_patch,
                        patch_dim=self.backbone.patch_repr_dim,
                        prop_dim=cond_dim,
                        proj_dim=spatial_emb,
                        dropout=dropout,
                    )
                if compress2:
                    self.compress2 = compress2
                else:
                    self.compress2 = deepcopy(self.compress1)

            else:  # TODO: clean up
                if compress:
                    self.compress = compress
                else:
                    self.compress = SpatialEmb(
                        num_patch=self.backbone.num_patch,
                        patch_dim=self.backbone.patch_repr_dim,
                        prop_dim=cond_dim,
                        proj_dim=spatial_emb,
                        dropout=dropout,
                    )
            visual_feature_dim = spatial_emb * num_img
        else:
            if compress:
                self.compress = compress
            else:
                self.compress = nn_Sequential([
                    nn_Linear(self.backbone.repr_dim, visual_feature_dim, name_Dense = "VisionDiffusionMLP_1"),
                    nn_LayerNorm(),
                    nn_Dropout(dropout),
                    nn_ReLU(),
                ])

        # diffusion
        input_dim = (
            time_dim + action_dim * horizon_steps + visual_feature_dim + cond_dim
        )

        output_dim = action_dim * horizon_steps
        
        if time_embedding:
            self.time_embedding = time_embedding
        else:
            self.time_embedding = nn_Sequential([
                SinusoidalPosEmb(time_dim),
                nn_Linear(time_dim, time_dim * 2, name_Dense = "VisionDiffusionMLP_time_embedding1"),
                nn_Mish(),
                nn_Linear(time_dim * 2, time_dim, name_Dense = "VisionDiffusionMLP_time_embedding2"),
            ])

        if residual_style:
            model = ResidualMLP
        else:
            model = MLP

        dim_list = [input_dim] + mlp_dims + [output_dim]

        print("dim_list = ", dim_list)

        if mlp_mean:
            self.mlp_mean = mlp_mean
        else:
            self.mlp_mean = model(
                dim_list,
                activation_type=activation_type,
                out_activation_type=out_activation_type,
                use_layernorm=use_layernorm,
            )

        self.time_dim = time_dim







    def get_config(self):

        if OUTPUT_FUNCTION_HEADER:
            print("VisionDiffusionMLP: get_config()")

        config = super(VisionDiffusionMLP, self).get_config()


        # print every property with its type and value
        if OUTPUT_VARIABLES:
            print("Checking VisionDiffusionMLP Config elements:")

            print(f"backbone: {self.backbone}, type: {type(self.backbone)}")

            print(f"action_dim: {self.action_dim}, type: {type(self.action_dim)}")
            print(f"horizon_steps: {self.horizon_steps}, type: {type(self.horizon_steps)}")
            print(f"cond_dim: {self.cond_dim}, type: {type(self.cond_dim)}")

            print(f"img_cond_steps: {self.img_cond_steps}, type: {type(self.img_cond_steps)}")
            print(f"time_dim: {self.time_dim}, type: {type(self.time_dim)}")

            print(f"mlp_dims: {self.mlp_dims}, type: {type(self.mlp_dims)}")
            print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")

            print(f"out_activation_type: {self.out_activation_type}, type: {type(self.out_activation_type)}")

            print(f"use_layernorm: {self.use_layernorm}, type: {type(self.use_layernorm)}")
            print(f"residual_style: {self.residual_style}, type: {type(self.residual_style)}")

            print(f"spatial_emb: {self.spatial_emb}, type: {type(self.spatial_emb)}")
            print(f"visual_feature_dim: {self.visual_feature_dim}, type: {type(self.visual_feature_dim)}")
            print(f"dropout: {self.dropout}, type: {type(self.dropout)}")

            print(f"num_img: {self.num_img}, type: {type(self.num_img)}")
            print(f"augment: {self.augment}, type: {type(self.augment)}")
        
        config.update({

            "action_dim": self.action_dim,
            "horizon_steps": self.horizon_steps,
            "cond_dim": self.cond_dim,

            "img_cond_steps": self.img_cond_steps,
            "time_dim": self.time_dim,

            "mlp_dims": self.mlp_dims,
            "activation_type": self.activation_type,
            "out_activation_type": self.out_activation_type,

            "use_layernorm": self.use_layernorm,
            "residual_style": self.residual_style,

            "spatial_emb": self.spatial_emb,
            "visual_feature_dim": self.visual_feature_dim,

            "dropout": self.dropout,
            "num_img": self.num_img,
            "augment": self.augment
        })




        if self.spatial_emb > 0:
            assert self.spatial_emb > 1, "this is the dimension"
            if self.num_img > 1:
                config.update({
                    "compress1": tf.keras.layers.serialize(self.compress1),
                    "compress2": tf.keras.layers.serialize(self.compress2),
                })                
            else:
                config.update({
                    "compress": tf.keras.layers.serialize(self.compress),
                })                
        else:
            config.update({
                "compress": tf.keras.layers.serialize(self.compress),
            })

        config.update({
            "time_embedding": tf.keras.layers.serialize(self.time_embedding),
            "mlp_mean": tf.keras.layers.serialize(self.mlp_mean),
            "backbone": tf.keras.layers.serialize(self.backbone),
        })


        return config



    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("VisionDiffusionMLP: from_config()")

        from tensorflow.keras.utils import get_custom_objects

        get_custom_objects().update(cur_dict)

        spatial_emb = config.pop("spatial_emb")
        num_img = config.pop("num_img")


        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
            if num_img > 1:
                compress = None
                compress1 = tf.keras.layers.deserialize(config.pop("compress1"),  custom_objects=get_custom_objects() )
                compress2 = tf.keras.layers.deserialize(config.pop("compress2"),  custom_objects=get_custom_objects() )
            else:
                compress = tf.keras.layers.deserialize(config.pop("compress"),  custom_objects=get_custom_objects() )
                compress1 = None
                compress2 = None
        else:
            compress = tf.keras.layers.deserialize(config.pop("compress"),  custom_objects=get_custom_objects() )
            compress1 = None
            compress2 = None

        time_embedding = tf.keras.layers.deserialize(config.pop("time_embedding"),  custom_objects=get_custom_objects() )
        mlp_mean = tf.keras.layers.deserialize(config.pop("mlp_mean"),  custom_objects=get_custom_objects() )
        backbone = tf.keras.layers.deserialize(config.pop("backbone"),  custom_objects=get_custom_objects() )


        result = cls(backbone=backbone, spatial_emb = spatial_emb, num_img = num_img, compress = compress, compress1 = compress1, compress2 = compress2, time_embedding = time_embedding, mlp_mean = mlp_mean, **config)

        return result





















    def call(self, inputs
            #  , **kwargs
             ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        """

        x, time, cond_state, cond_rgb = inputs

        if OUTPUT_FUNCTION_HEADER:
            print("mlp_diffusion.py: VisionDiffusionMLP.call()")

        B, Ta, Da = x.shape
        _, T_rgb, C, H, W = cond_rgb.shape

        # flatten chunk
        x = torch_tensor_view(x, [B, -1])

        # flatten history
        state = torch_tensor_view(cond_state, [B, -1])

        # Take recent images
        rgb = cond_rgb[:, -self.img_cond_steps:]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = torch_reshape(rgb, [B, T_rgb, self.num_img, 3, H, W])
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        # convert rgb to float32 for augmentation
        rgb = torch_tensor_float(rgb)

        # get vit output - pass in two images separately
        if self.num_img > 1:
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1(feat1, state)
            feat2 = self.compress2(feat2, state)
            feat = torch_cat([feat1, feat2], dim=-1)
        else:
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)

            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress.call(feat, state)
            else:
                feat = torch_flatten(feat, 1, -1)
                feat = self.compress(feat)

        cond_encoded = torch_cat([feat, state], dim=-1)

        # append time and cond
        time = torch_tensor_view(time, [B, 1])
        time_emb = torch_tensor_view( self.time_embedding(time), B, self.time_dim )
        x = torch_cat([x, time_emb, cond_encoded], dim=-1)

        # mlp
        out = self.mlp_mean(x)
        return torch_tensor_view(out, [B, Ta, Da])



































































@register_keras_serializable(package="Custom")
class DiffusionMLP(tf.keras.layers.Layer):
    
    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        time_embedding = None,
        cond_mlp = None,
        mlp_mean = None,
        name="DiffusionMLP",
        **kwargs
    ):
    
        if OUTPUT_FUNCTION_HEADER:
            print("mlp_diffusion.py: DiffusionMLP.__init__()")

        super(DiffusionMLP, self).__init__(**kwargs)

        if OUTPUT_POSITIONS:
            print("before sinusiodalPosEmb()")

        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.cond_dim = cond_dim

        if OUTPUT_VARIABLES:
            print("self.cond_dim = ", self.cond_dim)

        self.time_dim = time_dim
        
        self.mlp_dims = list(mlp_dims)

        if OUTPUT_VARIABLES:
            print("self.mlp_dims = ", self.mlp_dims)
            print( "type(self.mlp_dims) = ", type(self.mlp_dims) )

        if cond_mlp_dims != None:
            self.cond_mlp_dims = list(cond_mlp_dims)
        else:
            self.cond_mlp_dims = None

        self.activation_type = activation_type
        self.out_activation_type = out_activation_type
        self.use_layernorm = use_layernorm
        self.residual_style = residual_style


        self.output_dim = self.action_dim * self.horizon_steps

        if OUTPUT_VARIABLES:
            print("self.time_dim = ", self.time_dim)

        if time_embedding == None:
            self.time_embedding = nn_Sequential([
                SinusoidalPosEmb(time_dim),
                nn_Linear(time_dim, time_dim * 2, name_Dense = "DiffusionMLP_time_embedding_1"),
                nn_Mish(),
                nn_Linear(time_dim * 2, time_dim, name_Dense = "DiffusionMLP_time_embedding_2"),        
            ], name = "nn_Sequential_time_embedding")
        else:
            self.time_embedding = time_embedding




        if residual_style:
            model = ResidualMLP
        else:
            model = MLP

        

        if self.cond_mlp_dims is not None:

            if cond_mlp == None:
                self.cond_mlp = MLP(
                    [self.cond_dim] + self.cond_mlp_dims,
                    activation_type=self.activation_type,
                    out_activation_type="Identity",
                    name = "MLP_cond_mlp",
                )
            else:
                self.cond_mlp = cond_mlp

            self.input_dim = self.time_dim + self.action_dim * self.horizon_steps + self.cond_mlp_dims[-1]
        else:
            self.input_dim = self.time_dim + self.action_dim * self.horizon_steps + self.cond_dim
            self.cond_mlp = None



        if mlp_mean == None:
            self.mlp_mean = model(
                [self.input_dim] + self.mlp_dims + [self.output_dim],
                activation_type=self.activation_type,
                out_activation_type=self.out_activation_type,
                use_layernorm=self.use_layernorm,
                name = "mlp_mean",
            )
        else:
            self.mlp_mean = mlp_mean

        if OUTPUT_POSITIONS:
            print("after mlp_mean")
        
    def get_config(self):

        if OUTPUT_FUNCTION_HEADER:
            print("DiffusionMLP: get_config()")

        config = super(DiffusionMLP, self).get_config()

        # print every property with its type and value
        if OUTPUT_VARIABLES:
            print("Checking DiffusionMLP Config elements:")
            print(f"action_dim: {self.action_dim}, type: {type(self.action_dim)}")
            print(f"horizon_steps: {self.horizon_steps}, type: {type(self.horizon_steps)}")
            print(f"cond_dim: {self.cond_dim}, type: {type(self.cond_dim)}")
            print(f"time_dim: {self.time_dim}, type: {type(self.time_dim)}")
            
            print(f"mlp_dims: {self.mlp_dims}, type: {type(self.mlp_dims)}")


            print(f"cond_mlp_dims: {self.cond_mlp_dims}, type: {type(self.cond_mlp_dims)}")
            print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")
            print(f"out_activation_type: {self.out_activation_type}, type: {type(self.out_activation_type)}")
            print(f"use_layernorm: {self.use_layernorm}, type: {type(self.use_layernorm)}")
            print(f"residual_style: {self.residual_style}, type: {type(self.residual_style)}")
            print(f"output_dim: {self.output_dim}, type: {type(self.output_dim)}")
            print(f"input_dim: {self.input_dim}, type: {type(self.input_dim)}")
        
        config.update({
            "action_dim": self.action_dim,
            "horizon_steps": self.horizon_steps,
            "cond_dim": self.cond_dim,
            "time_dim": self.time_dim,
            "mlp_dims": self.mlp_dims,
            "cond_mlp_dims": self.cond_mlp_dims,
            "activation_type": self.activation_type,
            "out_activation_type": self.out_activation_type,
            "use_layernorm": self.use_layernorm,
            "residual_style": self.residual_style,
            
        })


        config.update({
            "time_embedding": 
            tf.keras.layers.serialize(self.time_embedding),
            "cond_mlp": 
            tf.keras.layers.serialize(self.cond_mlp),
            "mlp_mean": tf.keras.layers.serialize(self.mlp_mean),
        })

        return config



    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("DiffusionMLP: from_config()")

        from model.diffusion.mlp_diffusion import DiffusionMLP
        from model.diffusion.diffusion import DiffusionModel
        from model.common.mlp import MLP, ResidualMLP
        from model.diffusion.modules import SinusoidalPosEmb
        from model.common.modules import SpatialEmb, RandomShiftsAug
        from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish

        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
            'DiffusionMLP': DiffusionMLP,
            'SinusoidalPosEmb': SinusoidalPosEmb,   
            'MLP': MLP,                            # Custom MLP layer
            'ResidualMLP': ResidualMLP,            # Custom ResidualMLP layer
            'nn_Sequential': nn_Sequential,        # Custom Sequential class
            'nn_Linear': nn_Linear,
            'nn_LayerNorm': nn_LayerNorm,
            'nn_Dropout': nn_Dropout,
            'nn_ReLU': nn_ReLU,
            'nn_Mish': nn_Mish,
            'SpatialEmb': SpatialEmb,
            'RandomShiftsAug': RandomShiftsAug,
         }
        # Register custom class with Keras
        get_custom_objects().update(cur_dict)

        time_embedding = tf.keras.layers.deserialize(config.pop("time_embedding") ,  custom_objects=get_custom_objects() )

        config_cond_mlp = config.pop("cond_mlp")
        if config_cond_mlp:
            cond_mlp = tf.keras.layers.deserialize(config_cond_mlp,  custom_objects=get_custom_objects() )
        else:
            cond_mlp = None

        residual_style = config.pop("residual_style")

        mlp_mean = tf.keras.layers.deserialize(config.pop("mlp_mean") ,  custom_objects=get_custom_objects() )


        result = cls(residual_style = residual_style, time_embedding = time_embedding, cond_mlp = cond_mlp, mlp_mean = mlp_mean, **config)
        return result



    def call(
        self,
        inputs,
        training = True,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """


        if OUTPUT_FUNCTION_HEADER:
            print("mlp_diffusion.py: DiffusionMLP.call()", flush = True)


        x, time, cond_state = inputs


        B, Ta, Da = x.shape


        # assert 
        B = x.shape[0]
        
        
        assert Ta == self.horizon_steps
        assert Da == self.action_dim

        # flatten chunk
        x = torch_tensor_view(x, B, -1)

        if OUTPUT_VARIABLES:

            print("Diffusion_MLP: call(): x1 = ", x)

        # flatten history
        state = torch_tensor_view( cond_state, B, -1 )

        if OUTPUT_VARIABLES:
            print("Diffusion_MLP: call(): state = ", state)

        # obs encoder
        if hasattr(self, "cond_mlp") and self.cond_mlp:
            state = self.cond_mlp(state)

        # append time and cond
        time = torch_tensor_view(time, B, 1)

        if OUTPUT_VARIABLES:
            print("time = ", time)


        temp_result = self.time_embedding(time)

        if OUTPUT_VARIABLES:       
            print("temp_result = ", temp_result)

            print("temp_result.shape = ", temp_result.shape)


        time_emb = torch_tensor_view(temp_result, B, self.time_dim)


        if OUTPUT_VARIABLES:
            print("type(time_emb) = ", type(time_emb))

        if OUTPUT_VARIABLES:
            print("Diffusion_MLP: call(): time_emb = ", time_emb)


        x = torch_cat([x, time_emb, state], dim=-1)

        if OUTPUT_VARIABLES:
            print("Diffusion_MLP: call(): x2 = ", x)

        if OUTPUT_VARIABLES:
            print("self.mlp_mean = ", self.mlp_mean)

        # mlp head
        out = self.mlp_mean(x)


        if OUTPUT_VARIABLES:
            print("out = ", out)


        if OUTPUT_VARIABLES:
            print("DiffusionMLP call out.shape = ", out.shape)

        result = torch_tensor_view(out, B, Ta, Da)

        if OUTPUT_VARIABLES:
            print("DiffusionMLP call result.shape = ", result.shape)

        return result










    
