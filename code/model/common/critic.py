"""
Critic networks.

"""

from typing import Union

import tensorflow as tf
from tensorflow.keras import layers, Model

import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.common.modules import SpatialEmb, RandomShiftsAug


from util.torch_to_tf import torch_tensor_view, torch_cat, torch_squeeze, torch_reshape,\
torch_tensor_float\



from util.torch_to_tf import nn_TransformerEncoder, nn_TransformerEncoderLayer, nn_TransformerDecoder,\
nn_TransformerDecoderLayer, einops_layers_torch_Rearrange, nn_GroupNorm, nn_ConvTranspose1d, nn_Conv2d, nn_Conv1d, \
nn_MultiheadAttention, nn_LayerNorm, nn_Embedding, nn_ModuleList, nn_Sequential, \
nn_Linear, nn_Dropout, nn_ReLU, nn_GELU, nn_ELU, nn_Mish, nn_Softplus, nn_Identity, nn_Tanh
from model.diffusion.unet import ResidualBlock1D, Unet1D
from model.diffusion.modules import Conv1dBlock, Upsample1d, Downsample1d, SinusoidalPosEmb
from model.diffusion.eta import EtaStateAction, EtaState, EtaAction, EtaFixed
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
"Gaussian_Transformer": Gaussian_Transformer, 
"GMM_Transformer": GMM_Transformer, 
"Transformer": Transformer,
"SpatialEmb": SpatialEmb,
"RandomShiftsAug": RandomShiftsAug,
"MLP": MLP,
"ResidualMLP": ResidualMLP, 
"TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
}







class CriticObs(tf.keras.Model):
    """State-only critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        **kwargs,
    ):

        print("critic.py: CriticObs.__init__()")

        super().__init__()
        mlp_dims = [cond_dim] + mlp_dims + [1]
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )


    def call(self, cond):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            or (B, num_feature) from ViT encoder
        """

        print("critic.py: CriticObs.forward()")

        if isinstance(cond, dict):
            # B = len(cond["state"])
            B = tf.shape(cond["state"])[0]

            # flatten history
            state = torch_tensor_view(cond["state"], [B, -1])
        else:
            state = cond
        q1 = self.Q1(state)
        return q1





class CriticObsAct(tf.keras.Model):
    """State-action double critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        action_dim,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_tyle=False,
        double_q=True,
        Q1 = None,
        Q2 = None,
        **kwargs,
    ):
        self.cond_dim = cond_dim
        self.mlp_dims = list(mlp_dims)
        self.action_dim = action_dim
        self.action_steps = action_steps
        self.activation_type=activation_type
        self.use_layernorm=use_layernorm
        self.residual_tyle=residual_tyle
        self.double_q=double_q

        print("critic.py: CriticObsAct.__init__()")

        super().__init__()
        mlp_dims = [cond_dim + action_dim * action_steps] + mlp_dims + [1]
        if residual_tyle:
            model = ResidualMLP
        else:
            model = MLP

        if Q1:
            self.Q1 = Q1
        else:
            self.Q1 = model(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )

        if double_q:
            if Q2:
                self.Q2 = Q2
            else:
                self.Q2 = model(
                    mlp_dims,
                    activation_type=activation_type,
                    out_activation_type="Identity",
                    use_layernorm=use_layernorm,
                )


    def get_config(self):
        # print("CriticObsAct: get_config()")
        config = super(CriticObsAct, self).get_config()


        print(f"cond_dim: {self.cond_dim}, type: {type(self.cond_dim)}")
        print(f"mlp_dims: {self.mlp_dims}, type: {type(self.mlp_dims)}")
        print(f"action_dim: {self.action_dim}, type: {type(self.action_dim)}")
        print(f"action_steps: {self.action_steps}, type: {type(self.action_steps)}")
        print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")
        print(f"use_layernorm: {self.use_layernorm}, type: {type(self.use_layernorm)}")
        print(f"residual_tyle: {self.residual_tyle}, type: {type(self.residual_tyle)}")
        print(f"double_q: {self.double_q}, type: {type(self.double_q)}")

        print(f"Q1: {self.Q1}, type: {type(self.Q1)}")
    
        config.update({
            "Q1": 
            tf.keras.layers.serialize(self.Q1),
        })

        if hasattr(self, "Q2"):
            print(f"Q2: {self.Q2}, type: {type(self.Q2)}")

            config.update({
                "Q2": 
                tf.keras.layers.serialize(self.Q2),
            })
        else:
            config.update({
                "Q2": None
            })


        config.update({
            "cond_dim" : self.cond_dim,
            "mlp_dims" : self.mlp_dims,
            "action_dim" : self.action_dim,
            "action_steps" : self.action_steps,
            "activation_type" : self.activation_type,
            "use_layernorm" : self.use_layernorm,
            "residual_tyle" : self.residual_tyle,
            "double_q" : self.double_q
        })
    

        return config



    @classmethod
    def from_config(cls, config):
        print("DiffusionMLP: from_config()")
        from tensorflow.keras.utils import get_custom_objects

        # Register custom class with Keras
        get_custom_objects().update(cur_dict)
        Q1 = tf.keras.layers.deserialize(config.pop("Q1") ,  custom_objects=get_custom_objects() )
        
        config_Q2 = config.pop("Q2")
        if config_Q2:
            Q2 = tf.keras.layers.deserialize(config_Q2 ,  custom_objects=get_custom_objects() )
        else:
            Q2 = None

        result = cls(Q1 = Q1, Q2 = Q2, **config)
        return result





    def call(self, cond: dict, action):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        action: (B, Ta, Da)
        """

        print("critic.py: CriticObsAct.call()")

        B = tf.shape(cond["state"])[0]

        # flatten history
        state = torch_tensor_view(cond["state"], [B, -1])

        # flatten action
        action = torch_tensor_view(action, [B, -1])

        x = torch_cat((state, action), dim=-1)
        
        q1 = self.Q1(x)
        if hasattr(self, 'Q2'):
            q2 = self.Q2(x)
            return torch_squeeze(q1, 1), torch_squeeze(q2, 1)
        else:
            return torch_squeeze(q1, 1)


















class ViTCritic(CriticObs):
    """ViT + MLP, state only"""

    def __init__(
        self,
        backbone,
        cond_dim,
        img_cond_steps=1,
        spatial_emb=128,
        dropout=0,
        augment=False,
        num_img=1,
        **kwargs,
    ):

        print("critic.py: ViTCritic.__init__()")

        # update input dim to mlp
        mlp_obs_dim = spatial_emb * num_img + cond_dim
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)
        self.backbone = backbone
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if num_img > 1:
            self.compress1 = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            self.compress2 = deepcopy(self.compress1)
        else:  # TODO: clean up
            self.compress = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment

    def call(
        self,
        cond: dict,
        no_augment=False,
    ):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        no_augment: whether to skip augmentation

        TODO long term: more flexible handling of cond
        """

        print("critic.py: ViTCritic.call()")

        B, T_rgb, C, H, W = cond["rgb"].shape.as_list()

        # flatten history
        # state = cond["state"].view(B, -1)
        state = torch_tensor_view(cond["state"], B, -1)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps :]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = torch_reshape(rgb, [B, T_rgb, self.num_img, 3, H, W])
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        

        # convert rgb to float32 for augmentation
        rgb = torch_tensor_float( rgb )

        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment and not no_augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1(feat1, state)
            feat2 = self.compress2(feat2, state)
            feat = torch_cat([feat1, feat2], dim=-1)
        else:  # single image
            if self.augment and not no_augment:
                rgb = self.aug(rgb)  # uint8 -> float32
            feat = self.backbone(rgb)
            feat = self.compress.call(feat, state)
        feat = torch_cat([feat, state], axis=-1)
        return super().call(feat)


































class CriticObsAct_DACER(tf.keras.Model):
    """State-action double critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        action_dim,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_tyle=False,
        double_q=True,
        Q1 = None,
        Q2 = None,
        **kwargs,
    ):
        self.cond_dim = cond_dim
        self.mlp_dims = list(mlp_dims)
        self.action_dim = action_dim
        self.action_steps = action_steps
        self.activation_type=activation_type
        self.use_layernorm=use_layernorm
        self.residual_tyle=residual_tyle
        self.double_q=double_q

        print("critic.py: CriticObsAct.__init__()")

        super().__init__()
        mlp_dims = [cond_dim + action_dim * action_steps] + mlp_dims + [2]
        if residual_tyle:
            model = ResidualMLP
        else:
            model = MLP

        if Q1:
            self.Q1 = Q1
        else:
            self.Q1 = model(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )

        if double_q:
            if Q2:
                self.Q2 = Q2
            else:
                self.Q2 = model(
                    mlp_dims,
                    activation_type=activation_type,
                    out_activation_type="Identity",
                    use_layernorm=use_layernorm,
                )


    def get_config(self):
        print("CriticObsAct_DACER: get_config()")
        config = super(CriticObsAct_DACER, self).get_config()


        print(f"cond_dim: {self.cond_dim}, type: {type(self.cond_dim)}")
        print(f"mlp_dims: {self.mlp_dims}, type: {type(self.mlp_dims)}")
        print(f"action_dim: {self.action_dim}, type: {type(self.action_dim)}")
        print(f"action_steps: {self.action_steps}, type: {type(self.action_steps)}")
        print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")
        print(f"use_layernorm: {self.use_layernorm}, type: {type(self.use_layernorm)}")
        print(f"residual_tyle: {self.residual_tyle}, type: {type(self.residual_tyle)}")
        print(f"double_q: {self.double_q}, type: {type(self.double_q)}")

        print(f"Q1: {self.Q1}, type: {type(self.Q1)}")
        print(f"Q2: {self.Q2}, type: {type(self.Q2)}")


        config.update({
            "cond_dim" : self.cond_dim,
            "mlp_dims" : self.mlp_dims,
            "action_dim" : self.action_dim,
            "action_steps" : self.action_steps,
            "activation_type" : self.activation_type,
            "use_layernorm" : self.use_layernorm,
            "residual_tyle" : self.residual_tyle,
            "double_q" : self.double_q
        })
    
    
        config.update({
            "Q1": 
            tf.keras.layers.serialize(self.Q1),
            "Q2": 
            tf.keras.layers.serialize(self.Q2),
        })

        return config



    @classmethod
    def from_config(cls, config):
        print("CriticObsAct_DACER: from_config()")

        from tensorflow.keras.utils import get_custom_objects


        # Register custom class with Keras
        get_custom_objects().update(cur_dict)

        Q1 = tf.keras.layers.deserialize(config.pop("Q1") ,  custom_objects=get_custom_objects() )

        Q2 = tf.keras.layers.deserialize(config.pop("Q2") ,  custom_objects=get_custom_objects() )


        result = cls(Q1 = Q1, Q2 = Q2, **config)
        return result





    def call(self, cond: dict, action):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        action: (B, Ta, Da)
        """

        print("critic.py: CriticObsAct_DACER.call()")

        B = tf.shape(cond["state"])[0]

        # flatten history
        # state = tf.reshape(cond["state"], [B, -1])
        state = torch_tensor_view(cond["state"], [B, -1])

        # flatten action
        # action = tf.reshape(action, [B, -1])
        action = torch_tensor_view(action, [B, -1])

        # x = tf.concat([state, action], axis=-1)
        x = torch_cat((state, action), dim=-1)
        
        q1 = self.Q1(x)
        if hasattr(self, 'Q2'):
            q2 = self.Q2(x)

            q1_mean, q1_std = q1[..., 0], q1[..., 1]
            q2_mean, q2_std = q2[..., 0], q2[..., 1]

        # print("q1_mean.shape = ", q1_mean.shape)
        # print("q1_std.shape = ", q1_std.shape)
        # print("q2_mean.shape = ", q2_mean.shape)
        # print("q2_std.shape = ", q2_std.shape)

        #     # return torch_squeeze(q1, 1), torch_squeeze(q2, 1)
        #     return torch_squeeze(q1_mean, 1), torch_squeeze(q1_std, 1), torch_squeeze(q2_mean, 1), torch_squeeze(q2_std, 1)
        # else:
        #     # return torch_squeeze(q1, 1)
        #     return torch_squeeze(q1_mean, 1), torch_squeeze(q1_std, 1)

            return q1_mean, tf.math.softplus(q1_std), q2_mean, tf.math.softplus(q2_std)
        else:
            # return torch_squeeze(q1, 1)
            return q1_mean, tf.math.softplus(q1_std)





















