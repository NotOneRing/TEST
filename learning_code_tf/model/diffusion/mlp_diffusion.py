import tensorflow as tf
import numpy as np
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug

log = tf.get_logger()


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
    ):
        print("mlp_diffusion.py: VisionDiffusionMLP.__init__()", flush=True)

        super(VisionDiffusionMLP, self).__init__()

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
            self.compress = tf.keras.Sequential([
                tf.keras.layers.Dense(visual_feature_dim),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.ReLU(),
            ])

        # diffusion
        input_dim = time_dim + action_dim * horizon_steps + visual_feature_dim + cond_dim
        output_dim = action_dim * horizon_steps
        self.time_embedding = tf.keras.Sequential([
            SinusoidalPosEmb(time_dim),
            tf.keras.layers.Dense(time_dim * 2, activation="mish"),
            tf.keras.layers.Dense(time_dim),
        ])

        if residual_style:
            model = ResidualMLP
        else:
            model = MLP

        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def call(self, x, time, cond, **kwargs):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        """

        print("mlp_diffusion.py: VisionDiffusionMLP.call()", flush=True)

        B, Ta, Da = x.shape
        _, T_rgb, C, H, W = cond["rgb"].shape

        # flatten chunk
        x = tf.reshape(x, [B, -1])

        # flatten history
        state = tf.reshape(cond["state"], [B, -1])

        # Take recent images
        rgb = cond["rgb"][:, -self.img_cond_steps:]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = tf.reshape(rgb, [B, T_rgb, self.num_img, 3, H, W])
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        # convert rgb to float32 for augmentation
        rgb = tf.cast(rgb, tf.float32)

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
            feat = tf.concat([feat1, feat2], axis=-1)
        else:
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)

            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress(feat, state)
            else:
                feat = tf.reshape(feat, [B, -1])
                feat = self.compress(feat)

        cond_encoded = tf.concat([feat, state], axis=-1)

        # append time and cond
        time = tf.reshape(time, [B, 1])
        time_emb = self.time_embedding(time)
        x = tf.concat([x, time_emb, cond_encoded], axis=-1)

        # mlp
        out = self.mlp_mean(x)
        return tf.reshape(out, [B, Ta, Da])




class DiffusionMLP(tf.keras.Model):

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
    ):
        print("mlp_diffusion.py: DiffusionMLP.__init__()", flush=True)

        super(DiffusionMLP, self).__init__()

        output_dim = action_dim * horizon_steps
        self.time_embedding = tf.keras.Sequential([
            SinusoidalPosEmb(time_dim),
            tf.keras.layers.Dense(time_dim * 2, activation="mish"),
            tf.keras.layers.Dense(time_dim),
        ])

        if residual_style:
            model = ResidualMLP
        else:
            model = MLP

        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            input_dim = time_dim + action_dim * horizon_steps + cond_mlp_dims[-1]
        else:
            input_dim = time_dim + action_dim * horizon_steps + cond_dim

        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def call(self, x, time, cond, **kwargs):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """

        print("mlp_diffusion.py: DiffusionMLP.call()", flush=True)

        B, Ta, Da = x.shape

        # flatten chunk
        x = tf.reshape(x, [B, -1])

        # flatten history
        state = tf.reshape(cond["state"], [B, -1])

        # obs encoder
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        # append time and cond
        time = tf.reshape(time, [B, 1])
        time_emb = self.time_embedding(time)
        x = tf.concat([x, time_emb, state], axis=-1)

        # mlp head
        out = self.mlp_mean(x)
        return tf.reshape(out, [B, Ta, Da])









