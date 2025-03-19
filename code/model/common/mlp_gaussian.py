"""
MLP models for Gaussian policy.

"""

import tensorflow as tf
import numpy as np

import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.common.modules import SpatialEmb, RandomShiftsAug


from util.torch_to_tf import nn_Linear, nn_Sequential, nn_LayerNorm,\
nn_Dropout, nn_ReLU, nn_Parameter, torch_log, torch_tensor, torch_tensor_view,\
torch_reshape, torch_flatten, torch_tensor_float, torch_cat, torch_tensor_repeat, torch_tanh, \
torch_clamp, torch_exp, torch_ones_like, load_tf_Variable, save_tf_Variable



from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, OUTPUT_POSITIONS, OUTPUT_FUNCTION_HEADER, DATASET_NAME






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
}








class Gaussian_VisionMLP(tf.keras.layers.Layer):
    """With ViT backbone"""

    def __init__(
        self,
        backbone,
        action_dim,
        horizon_steps,
        cond_dim,
        img_cond_steps=1,
        mlp_dims=[256, 256, 256],
        activation_type="Mish",
        residual_style=False,
        use_layernorm=False,
        fixed_std=None,
        learn_fixed_std=False,
        std_min=0.01,
        std_max=1,
        spatial_emb=0,
        visual_feature_dim=128,
        dropout=0,
        num_img=1,
        augment=False,

        mlp_logvar = None,
        mlp_mean = None,
        logvar = None,
        logvar_min = None,
        logvar_max = None,
        compress = None, 
        compress1 = None,
        compress2 = None,
        **kwargs
    ):

        self.cond_dim = cond_dim
        self.mlp_dims = list(mlp_dims)
        self.activation_type = activation_type
        self.residual_style = residual_style
        self.use_layernorm = use_layernorm
        self.std_min = std_min
        self.std_max = std_max
        self.spatial_emb = spatial_emb
        self.visual_feature_dim = visual_feature_dim
        self.dropout = dropout


        print("mlp_gaussian.py: Gaussian_VisionMLP.__init__()")

        super().__init__()

        self.backbone = backbone
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
                    nn_Linear(self.backbone.repr_dim, visual_feature_dim),
                    nn_LayerNorm(visual_feature_dim),
                    nn_Dropout(dropout),
                    nn_ReLU(),
                ])

        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        input_dim = visual_feature_dim + cond_dim
        output_dim = action_dim * horizon_steps
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
                self.mlp_logvar = MLP(
                    [input_dim] + mlp_dims[-1:] + [output_dim],
                    activation_type=activation_type,
                    out_activation_type="Identity",
                    use_layernorm=use_layernorm,
                )
        elif learn_fixed_std:  # initialize to fixed_std
            if logvar:
                self.logvar = logvar
            else:
                self.logvar = nn_Parameter(
                    torch_log(torch_tensor( np.array([fixed_std**2 for _ in range(action_dim)]) )),
                    requires_grad=True,
                )


        if logvar_min:
            self.logvar_min = logvar_min
        else:
            self.logvar_min = nn_Parameter(
                torch_log(torch_tensor( np.array( [std_min**2] ) )), requires_grad=False
            )
            self.logvar_min = tf.cast(self.logvar_min, tf.float32)

        if logvar_max:
            self.logvar_max = logvar_max
            self.logvar_max = tf.cast(self.logvar_max, tf.float32)
        else:
            self.logvar_max = nn_Parameter(
                torch_log(torch_tensor( np.array( [std_max**2] ) )), requires_grad=False
            )


        self.use_fixed_std = fixed_std is not None
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std



    def get_config(self):

        # if OUTPUT_FUNCTION_HEADER:
        print("Gaussian_VisionMLP: get_config()")

        config = super(Gaussian_VisionMLP, self).get_config()


        # print every property with its type and value
        if OUTPUT_VARIABLES:
            print("Checking Gaussian_VisionMLP Config elements:")
            print(f"backbone: {self.backbone}, type: {type(self.backbone)}")
            print(f"action_dim: {self.action_dim}, type: {type(self.action_dim)}")
            print(f"horizon_steps: {self.horizon_steps}, type: {type(self.horizon_steps)}")
            print(f"cond_dim: {self.cond_dim}, type: {type(self.cond_dim)}")
            print(f"img_cond_steps: {self.img_cond_steps}, type: {type(self.img_cond_steps)}")
            print(f"mlp_dims: {self.mlp_dims}, type: {type(self.mlp_dims)}")
            print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")
            print(f"residual_style: {self.residual_style}, type: {type(self.residual_style)}")
            print(f"use_layernorm: {self.use_layernorm}, type: {type(self.use_layernorm)}")

            print(f"fixed_std: {self.fixed_std}, type: {type(self.fixed_std)}")
            print(f"learn_fixed_std: {self.learn_fixed_std}, type: {type(self.learn_fixed_std)}")

            print(f"std_min: {self.std_min}, type: {type(self.std_min)}")
            print(f"std_max: {self.std_max}, type: {type(self.std_max)}")

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
            "mlp_dims": self.mlp_dims,
            "activation_type": self.activation_type,
            "residual_style": self.residual_style,
            "use_layernorm": self.use_layernorm,
            "fixed_std": self.fixed_std,
            "learn_fixed_std": self.learn_fixed_std,
            "std_min": self.std_min,
            "std_max": self.std_max,

            "spatial_emb": self.spatial_emb,
            "visual_feature_dim": self.visual_feature_dim,
            "dropout": self.dropout,
            "num_img": self.num_img,
            "augment": self.augment,
        })


        if self.fixed_std is None:
            config.update({
                "mlp_logvar": tf.keras.layers.serialize(self.mlp_logvar),
            })

        elif self.learn_fixed_std:  # initialize to fixed_std
            save_tf_Variable(self.logvar, "Gaussian_VisionMLP_logvar" + DATASET_NAME)



        save_tf_Variable(self.logvar_min, "Gaussian_VisionMLP_logvar_min" + DATASET_NAME)
        save_tf_Variable(self.logvar_max, "Gaussian_VisionMLP_logvar_max" + DATASET_NAME)
        

        config.update({
            "mlp_mean": tf.keras.layers.serialize(self.mlp_mean),
        })

        config.update({
            "backbone": tf.keras.layers.serialize(self.backbone),
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

        return config



    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("Gaussian_MLP: from_config()")


        from tensorflow.keras.utils import get_custom_objects

        get_custom_objects().update(cur_dict)


        fixed_std = config.pop("fixed_std")

        learn_fixed_std = config.pop("learn_fixed_std")

        if not fixed_std:
            logvar = None
            mlp_logvar = tf.keras.layers.deserialize(config.pop("mlp_logvar"),  custom_objects=get_custom_objects() )
        elif learn_fixed_std:
            logvar = load_tf_Variable("Gaussian_VisionMLP_logvar" + DATASET_NAME)
            mlp_logvar = None
        else:
            logvar = None
            mlp_logvar = None



        logvar_min = load_tf_Variable("Gaussian_VisionMLP_logvar_min" + DATASET_NAME)
        logvar_max = load_tf_Variable("Gaussian_VisionMLP_logvar_max" + DATASET_NAME)
        

        mlp_mean = tf.keras.layers.deserialize(config.pop("mlp_mean") ,  custom_objects=get_custom_objects() )

        backbone = tf.keras.layers.deserialize(config.pop("backbone") ,  custom_objects=get_custom_objects() )

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

        result = cls(backbone = backbone, fixed_std = fixed_std, mlp_logvar = mlp_logvar, mlp_mean = mlp_mean, spatial_emb = spatial_emb, num_img = num_img, compress = compress, compress1 = compress1, compress2 = compress2, logvar = logvar, logvar_min = logvar_min, logvar_max = logvar_max, **config)

        return result





    def call(self, cond):

        print("mlp_gaussian.py: Gaussian_VisionMLP.call()")

        B = cond["rgb"].shape[0]

        _, T_rgb, C, H, W = cond["rgb"].shape

        # flatten history
        state = torch_tensor_view(cond["state"], (B, -1))

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps :]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = torch_reshape(rgb, (B, T_rgb, self.num_img, 3, H, W))
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        # convert rgb to float32 for augmentation
        rgb = torch_tensor_float(rgb)

        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
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
        else:  # single image
            if self.augment:
                rgb = self.aug(rgb)  # uint8 -> float32
            feat = self.backbone(rgb)

            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress(feat, state)  # Assuming the `SpatialEmb` class has a compatible `__call__` method
            else:
                feat = torch_flatten(feat, [1, -1])  # Flatten the feature map (assuming B is batch size)
                feat = self.compress(feat)  # Apply the MLP or other operations
        

        # MLP forward pass
        x_encoded = torch_cat([feat, state], dim=-1)
        out_mean = self.mlp_mean(x_encoded)
        out_mean = torch_tanh(out_mean)
        out_mean = torch_tensor_view(out_mean, (B, self.horizon_steps * self.action_dim))
        # tanh squashing in [-1, 1]

        # Handling scale (std)
        if self.learn_fixed_std:
            out_logvar = torch_clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch_exp(0.5 * out_logvar)
            out_scale = torch_tensor_view(out_scale, (1, self.action_dim))
            out_scale = torch_tensor_repeat(out_scale, [B, self.horizon_steps])
        elif self.use_fixed_std:
            out_scale = torch_ones_like(out_mean) * self.fixed_std
        else:
            out_logvar = self.mlp_logvar(x_encoded)
            out_logvar = torch_tensor_view(out_logvar, (B, self.horizon_steps * self.action_dim))
            out_logvar = torch_clamp(out_logvar, self.logvar_min, self.logvar_max)
            out_scale = torch_exp(0.5 * out_logvar)

        return out_mean, out_scale








class Gaussian_MLP(tf.keras.Model):
    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        mlp_dims=[256, 256, 256],
        activation_type="Mish",
        tanh_output=True,  # sometimes we want to apply tanh after sampling instead of here, e.g., in SAC
        residual_style=False,
        use_layernorm=False,
        dropout=0.0,
        fixed_std=None,
        learn_fixed_std=False,
        std_min=0.01,
        std_max=1,
        mlp_base = None, 
        mlp_logvar = None, 
        mlp_mean = None,
        **kwargs
    ):

        print("mlp_gaussian.py: Gaussian_MLP.__init__()")

        super().__init__()


        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.cond_dim = cond_dim
        self.mlp_dims = list(mlp_dims)
        self.activation_type = activation_type
        self.residual_style = residual_style
        self.use_layernorm = use_layernorm
        self.dropout = dropout
        self.std_min = std_min
        self.std_max = std_max


        input_dim = cond_dim
        output_dim = action_dim * horizon_steps
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        if fixed_std is None:
            # learning std
            if mlp_base:
                self.mlp_base = mlp_base
            else:
                print("mlp_base = ", mlp_base)
                print("[input_dim] + mlp_dims = ", [input_dim] + mlp_dims)
                self.mlp_base = model(
                    [input_dim] + mlp_dims,
                    activation_type=activation_type,
                    out_activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    use_layernorm_final=use_layernorm,
                    dropout=dropout,
                )
            if mlp_mean:
                self.mlp_mean = mlp_mean
            else:
                self.mlp_mean = MLP(
                    mlp_dims[-1:] + [output_dim],
                    out_activation_type="Identity",
                )
            if mlp_logvar:
                self.mlp_logvar = mlp_logvar
            else:
                self.mlp_logvar = MLP(
                    mlp_dims[-1:] + [output_dim],
                    out_activation_type="Identity",
                )
        else:
            if mlp_mean:
                self.mlp_mean = mlp_mean
            else:
                # no separate head for mean and std
                self.mlp_mean = model(
                    [input_dim] + mlp_dims + [output_dim],
                    activation_type=activation_type,
                    out_activation_type="Identity",
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )

            if learn_fixed_std:
                # initialize to fixed_std
                self.logvar = nn_Parameter(
                    torch_log(torch_tensor( np.array([fixed_std**2 for _ in range(action_dim)]) )),
                    requires_grad=True,
                )
                        
        self.logvar_min = nn_Parameter(
            torch_log(torch_tensor( np.array( [std_min**2] ) )), requires_grad=False
        )
        self.logvar_max = nn_Parameter(
            torch_log(torch_tensor( np.array( [std_max**2] ) )), requires_grad=False
        )

        self.logvar_min = tf.cast(self.logvar_min, tf.float32)
        self.logvar_max = tf.cast(self.logvar_max, tf.float32)


        self.use_fixed_std = fixed_std is not None
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std
        self.tanh_output = tanh_output



    def get_config(self):

        if OUTPUT_FUNCTION_HEADER:
            print("Gaussian_MLP: get_config()")

        config = super(Gaussian_MLP, self).get_config()





        # print every property with its type and value
        if OUTPUT_VARIABLES:
            print("Checking DiffusionMLP Config elements:")
            print(f"action_dim: {self.action_dim}, type: {type(self.action_dim)}")
            print(f"horizon_steps: {self.horizon_steps}, type: {type(self.horizon_steps)}")
            print(f"cond_dim: {self.cond_dim}, type: {type(self.cond_dim)}")
            print(f"mlp_dims: {self.mlp_dims}, type: {type(self.mlp_dims)}")
            print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")
            print(f"tanh_output: {self.tanh_output}, type: {type(self.tanh_output)}")
            print(f"residual_style: {self.residual_style}, type: {type(self.residual_style)}")
            print(f"use_layernorm: {self.use_layernorm}, type: {type(self.use_layernorm)}")
            print(f"dropout: {self.dropout}, type: {type(self.dropout)}")
            print(f"fixed_std: {self.fixed_std}, type: {type(self.fixed_std)}")
            print(f"learn_fixed_std: {self.learn_fixed_std}, type: {type(self.learn_fixed_std)}")
            print(f"std_min: {self.std_min}, type: {type(self.std_min)}")
            print(f"std_max: {self.std_max}, type: {type(self.std_max)}")
        
        config.update({
            "action_dim": self.action_dim,
            "horizon_steps": self.horizon_steps,
            "cond_dim": self.cond_dim,
            "mlp_dims": self.mlp_dims,
            "activation_type": self.activation_type,
            "tanh_output": self.tanh_output,
            "residual_style": self.residual_style,
            "use_layernorm": self.use_layernorm,
            "dropout": self.dropout,
            "fixed_std": self.fixed_std,
            "learn_fixed_std": self.learn_fixed_std,
            "std_min": self.std_min,
            "std_max": self.std_max
        })


        if self.fixed_std is None:
            config.update({
                "mlp_base": tf.keras.layers.serialize(self.mlp_base),
                "mlp_logvar": tf.keras.layers.serialize(self.mlp_logvar),
            })
            

        config.update({
            "mlp_mean": tf.keras.layers.serialize(self.mlp_mean),
        })


        return config



    @classmethod
    def from_config(cls, config):
        if OUTPUT_FUNCTION_HEADER:
            print("Gaussian_MLP: from_config()")



        from tensorflow.keras.utils import get_custom_objects

        # Register custom class with Keras
        get_custom_objects().update(cur_dict)



        fixed_std = config.pop("fixed_std")
        if not fixed_std:
            mlp_base = tf.keras.layers.deserialize(config.pop("mlp_base"),  custom_objects=get_custom_objects() )
            mlp_logvar = tf.keras.layers.deserialize(config.pop("mlp_logvar"),  custom_objects=get_custom_objects() )
        else:
            mlp_base = None
            mlp_logvar = None

        mlp_mean = tf.keras.layers.deserialize(config.pop("mlp_mean") ,  custom_objects=get_custom_objects() )

        result = cls(mlp_base = mlp_base, mlp_logvar = mlp_logvar, mlp_mean = mlp_mean, fixed_std = fixed_std, **config)

        return result



















    def call(self, cond):

        print("mlp_gaussian.py: Gaussian_MLP.call()")

        B = cond["state"].shape[0]

        # flatten history
        state = torch_tensor_view(cond["state"], (B, -1) )

        # mlp
        if hasattr(self, "mlp_base"):
            state = self.mlp_base(state)


        # Mean prediction
        out_mean = self.mlp_mean(state)
            
        if self.tanh_output:
            out_mean = torch_tanh(out_mean)

        out_mean = torch_tensor_view(out_mean, [B, self.horizon_steps * self.action_dim])

        # Standard deviation prediction
        if self.learn_fixed_std:
            out_logvar = torch_clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch_exp(0.5 * out_logvar)
            out_scale = torch_tensor_view(out_scale, [1, self.action_dim])
            out_scale = torch_tensor_repeat(out_scale, [B, self.horizon_steps])
        elif self.use_fixed_std:
            out_scale = torch_ones_like(out_mean) * self.fixed_std
        else:
            out_logvar = self.mlp_logvar(state)
            out_logvar = torch_tensor_view(out_logvar, [B, self.horizon_steps * self.action_dim])
            out_logvar = torch_tanh(out_logvar)




            # Scale to range [logvar_min, logvar_max]
            out_logvar = self.logvar_min + 0.5 * (self.logvar_max - self.logvar_min) * (out_logvar + 1)
            out_scale = torch_exp(0.5 * out_logvar)

        return out_mean, out_scale

    
