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
torch_reshape, torch_flatten, torch_tensor_float, torch_cat, torch_tensor_repeat,\



class Gaussian_VisionMLP(tf.keras.Model):
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
    ):

        print("mlp_gaussian.py: Gaussian_VisionMLP.__init__()")

        super().__init__()

        # vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
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
            visual_feature_dim = spatial_emb * num_img
        else:

            self.compress = nn_Sequential([
                nn_Linear(self.backbone.repr_dim, visual_feature_dim),
                nn_LayerNorm(visual_feature_dim),
                nn_Dropout(dropout),
                nn_ReLU(),
            ])

        # head
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        input_dim = visual_feature_dim + cond_dim
        output_dim = action_dim * horizon_steps
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
        if fixed_std is None:
            self.mlp_logvar = MLP(
                [input_dim] + mlp_dims[-1:] + [output_dim],
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )
        elif learn_fixed_std:  # initialize to fixed_std
            self.logvar = nn_Parameter(
                torch_log(torch_tensor([fixed_std**2 for _ in range(action_dim)])),
                requires_grad=True,
            )


        self.logvar_min = nn_Parameter(
            torch_log(torch_tensor(std_min**2)), requires_grad=False
        )
        self.logvar_max = nn_Parameter(
            torch_log(torch_tensor(std_max**2)), requires_grad=False
        )


        self.use_fixed_std = fixed_std is not None
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std


    def call(self, cond):

        print("mlp_gaussian.py: Gaussian_VisionMLP.call()")

        B = tf.shape(cond["rgb"])[0]

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
    ):

        print("mlp_gaussian.py: Gaussian_MLP.__init__()")

        super().__init__()
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        input_dim = cond_dim
        output_dim = action_dim * horizon_steps
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        if fixed_std is None:
            # learning std
            self.mlp_base = model(
                [input_dim] + mlp_dims,
                activation_type=activation_type,
                out_activation_type=activation_type,
                use_layernorm=use_layernorm,
                use_layernorm_final=use_layernorm,
                dropout=dropout,
            )
            self.mlp_mean = MLP(
                mlp_dims[-1:] + [output_dim],
                out_activation_type="Identity",
            )
            self.mlp_logvar = MLP(
                mlp_dims[-1:] + [output_dim],
                out_activation_type="Identity",
            )
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
                    torch_log(torch_tensor([fixed_std**2 for _ in range(action_dim)])),
                    requires_grad=True,
                )
                        
        self.logvar_min = nn_Parameter(
            torch_log(torch_tensor(std_min**2)), requires_grad=False
        )
        self.logvar_max = nn_Parameter(
            torch_log(torch_tensor(std_max**2)), requires_grad=False
        )

        self.use_fixed_std = fixed_std is not None
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std
        self.tanh_output = tanh_output


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

    
