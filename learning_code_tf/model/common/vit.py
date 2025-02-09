"""
ViT image encoder implementation from IBRL, https://github.com/hengyuan-hu/ibrl
"""

from dataclasses import dataclass
from typing import List
import einops


import tensorflow as tf
from tensorflow.keras import layers, Model

import math

from util.torch_to_tf import nn_GELU, torch_flatten, nn_Conv2d, nn_GroupNorm, \
nn_Linear, nn_LayerNorm, nn_Dropout, torch_rand, torch_zeros, nn_Sequential, \
nn_Parameter, nn_ReLU, torch_nn_init_trunc_normal_, nn_Identity, torch_nn_init_zeros_


@dataclass
class VitEncoderConfig:
    patch_size: int = 8
    depth: int = 1
    embed_dim: int = 128
    num_heads: int = 4
    # act_layer = nn.GELU
    act_layer= nn_GELU,
    stride: int = -1
    embed_style: str = "embed2"
    embed_norm: int = 0



class VitEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        obs_shape: List[int],
        cfg: VitEncoderConfig,
        num_channel=3,
        img_h=96,
        img_w=96,
    ):

        print("vit.py: VitEncoder.__init__()")

        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        self.vit = MinVit(
            embed_style=cfg.embed_style,
            embed_dim=cfg.embed_dim,
            embed_norm=cfg.embed_norm,
            num_head=cfg.num_heads,
            depth=cfg.depth,
            num_channel=num_channel,
            img_h=img_h,
            img_w=img_w,
        )
        self.img_h = img_h
        self.img_w = img_w
        self.num_patch = self.vit.num_patches
        self.patch_repr_dim = self.cfg.embed_dim
        self.repr_dim = self.cfg.embed_dim * self.vit.num_patches

    def call(self, obs, flatten=False):
        print("vit.py: VitEncoder.call()")

        # assert obs.max() > 5
        obs = obs / 255.0 - 0.5
        feats = self.vit(obs)
        if flatten:
            # assert len(feats.shape) == 3, f"Expected feats to be 3D, but got {len(feats.shape)}D"
            feats = torch_flatten(feats, 1, 2)

        return feats


# class PatchEmbed1(nn.Module):
class PatchEmbed1(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_channel=3, img_h=96, img_w=96):

        print("vit.py: PatchEmbed1.__init__()")

        super().__init__()
        # self.conv = nn.Conv2d(num_channel, embed_dim, kernel_size=8, stride=8)
        # 输入维度是num_channel
        # self.conv = layers.Conv2D(embed_dim, kernel_size=8, strides=8, padding="valid")

        self.conv = nn_Conv2d(num_channel, embed_dim, kernel_size=8, stride=8)

        self.num_patch = math.ceil(img_h / 8) * math.ceil(img_w / 8)
        self.patch_dim = embed_dim

    def call(self, x):

        print("vit.py: PatchEmbed1.call()")

        y = self.conv(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c")

        return y


class PatchEmbed2(tf.keras.layers.Layer):
    def __init__(self, embed_dim, use_norm, num_channel=3, img_h=96, img_w=96):

        print("vit.py: PatchEmbed2.__init__()")

        super().__init__()

        #输入是num_channel
        layers = [
            nn_Conv2d(num_channel, embed_dim, kernel_size=8, stride=4),
            nn_GroupNorm() if use_norm else nn_Identity(),
            nn_ReLU(),
            nn_Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2),
        ]

        self.embed = nn_Sequential(layers)

        H1 = math.ceil((img_h - 8) / 4) + 1
        W1 = math.ceil((img_w - 8) / 4) + 1
        H2 = math.ceil((H1 - 3) / 2) + 1
        W2 = math.ceil((W1 - 3) / 2) + 1
        self.num_patch = H2 * W2
        self.patch_dim = embed_dim

    def call(self, x):

        print("vit.py: PatchEmbed2.call()")

        y = self.embed(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c")
        return y


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_head):

        print("vit.py: MultiHeadAttention.__init__()")

        super().__init__()
        assert embed_dim % num_head == 0

        self.num_head = num_head
        #输入维度是embed_dim
        self.qkv_proj = nn_Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn_Linear(embed_dim, embed_dim)


    def call(self, x, attn_mask):
        """
        x: [batch, seq, embed_dim]
        """

        print("vit.py: MultiHeadAttention.call()")

        qkv = self.qkv_proj(x)

        # Split qkv into q, k, v
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)

        # Reshape for multi-head attention: [batch, seq_len, num_heads, embed_dim // num_heads]
        q = tf.reshape(q, (q.shape[0], q.shape[1], self.num_head, q.shape[-1] // self.num_head))
        k = tf.reshape(k, (k.shape[0], k.shape[1], self.num_head, k.shape[-1] // self.num_head))
        v = tf.reshape(v, (v.shape[0], v.shape[1], self.num_head, v.shape[-1] // self.num_head))

        # Transpose to shape: [batch, num_heads, seq_len, embed_dim // num_heads]
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # Scaled Dot-Product Attention
        attn_weights = tf.matmul(q, k, transpose_b=True)  # [batch, num_heads, seq_len, seq_len]
        attn_weights = attn_weights / tf.math.sqrt(tf.cast(q.shape[-1], tf.float32))  # Scaling

        if attn_mask is not None:
            attn_weights += (attn_mask * -1e9)  # Applying the attention mask

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)  # Softmax over the last dimension

        # Attention output
        attn_output = tf.matmul(attn_weights, v)  # [batch, num_heads, seq_len, embed_dim // num_heads]

        # Reshape back: [batch, seq_len, num_heads * embed_dim // num_heads]
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (attn_output.shape[0], attn_output.shape[1], -1))

        # Apply output projection
        return self.out_proj(attn_output)



class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_head, dropout):

        print("vit.py: TransformerLayer.__init__()")

        super().__init__()

        # #输入是embed_dim维度的
        # self.layer_norm1 = layers.LayerNormalization()
        # self.mha = MultiHeadAttention(embed_dim, num_head)
        # #输入是embed_dim维度的
        # self.layer_norm2 = layers.LayerNormalization()
        # #输入是embed_dim维度的
        # self.linear1 = layers.Dense(4 * embed_dim)
        # self.linear2 = layers.Dense(embed_dim)
        # self.dropout = layers.Dropout(dropout)

        self.layer_norm1 = nn_LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_head)

        self.layer_norm2 = nn_LayerNorm(embed_dim)
        self.linear1 = nn_Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn_Linear(4 * embed_dim, embed_dim)
        self.dropout = nn_Dropout(dropout)

    def call(self, x, attn_mask=None):

        print("vit.py: TransformerLayer.call()")

        x = x + self.dropout(self.mha(self.layer_norm1(x), attn_mask))
        x = x + self.dropout(self._ff_block(self.layer_norm2(x)))
        return x

    def _ff_block(self, x):

        print("vit.py: TransformerLayer._ff_block()")

        x = self.linear2(tf.nn.gelu(self.linear1(x)))
        return x


class MinVit(tf.keras.Model):
    def __init__(
        self,
        embed_style,
        embed_dim,
        embed_norm,
        num_head,
        depth,
        num_channel=3,
        img_h=96,
        img_w=96,
    ):

        print("vit.py: MinVit.__init__()")

        super().__init__()

        if embed_style == "embed1":
            self.patch_embed = PatchEmbed1(
                embed_dim,
                num_channel=num_channel,
                img_h=img_h,
                img_w=img_w,
            )
        elif embed_style == "embed2":
            self.patch_embed = PatchEmbed2(
                embed_dim,
                use_norm=embed_norm,
                num_channel=num_channel,
                img_h=img_h,
                img_w=img_w,
            )
        else:
            assert False


        # self.pos_embed = tf.Variable(tf.random.truncated_normal([1, self.patch_embed.num_patch, embed_dim], stddev=0.02))


        # layers = [TransformerLayer(embed_dim, num_head, dropout=0) for _ in range(depth)]
        # self.net = tf.keras.Sequential(*layers)

        # #输入维度是embed_dim
        # self.norm = layers.LayerNormalization()

        # self.num_patches = self.patch_embed.num_patch

        self.pos_embed = nn_Parameter(
            torch_zeros(1, self.patch_embed.num_patch, embed_dim)
        )
        layers = [
            TransformerLayer(embed_dim, num_head, dropout=0) for _ in range(depth)
        ]

        self.net = nn_Sequential(*layers)
        self.norm = nn_LayerNorm(embed_dim)
        self.num_patches = self.patch_embed.num_patch

        # weight init
        torch_nn_init_trunc_normal_(self.pos_embed, std=0.02)

        named_apply(init_weights_vit_timm, self)


    def call(self, x):

        print("vit.py: MinVit.call()")

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.net(x)
        return self.norm(x)






# def init_weights_vit_timm(module, name=""):
#     """ViT weight initialization, similar to timm for reproducibility"""

#     print("vit.py: init_weights_vit_timm()")

#     if isinstance(module, layers.Dense):
#         initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
#         module.kernel_initializer = initializer
#         if module.bias is not None:
#             module.bias_initializer = tf.keras.initializers.Zeros()
#     elif isinstance(module, layers.Conv2D):
#         initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
#         module.kernel_initializer = initializer


def init_weights_vit_timm(module, name: str = ""):
    if isinstance(module, nn_Linear):
        torch_nn_init_trunc_normal_(module.kernel, std=0.02)
        if module.bias is not None:
            torch_nn_init_zeros_(module.bias)




def named_apply(fn, module, name="", depth_first=True, include_root=False):
    """Recursively apply a function to a model's layers"""
    print("vit.py: named_apply()")
    if not depth_first and include_root:
        if isinstance(module, tf.keras.layers.Layer):
            fn(module, name=name)

    if hasattr(module, 'layers'):
        for i, child_layer in enumerate(module.layers):
            child_name = f"{name}.{child_layer.name}" if name else child_layer.name

            named_apply(
                fn=fn,
                module=child_layer,
                name=child_name,
                depth_first=depth_first,
                include_root=True,
            )

    if depth_first and include_root:
        if isinstance(module, tf.keras.layers.Layer):
            fn(module, name=name)
    return module



def test_patch_embed():
    print("vit.py: test_patch_embed()")

    # 测试第一个 PatchEmbed 类
    print("embed 1")
    embed = PatchEmbed1(128) 
    # x = tf.random.uniform([10, 96, 96, 3])  # 输入数据形状为 NHWC
    x = torch_rand(10, 3, 96, 96)
    y = embed(x)
    print("Output shape for embed 1:", y.shape)

    # 测试第二个 PatchEmbed 类
    print("embed 2")
    embed = PatchEmbed2(128)  # 对应第二种设置
    # x = tf.random.uniform([10, 96, 96, 3])  # 输入数据形状为 NHWC
    x = torch_rand(10, 3, 96, 96)
    y = embed(x)
    print("Output shape for embed 2:", y.shape)







def test_transformer_layer():
    print("Testing TransformerLayer...")
    # x = tf.random.uniform([10, 96, 96, 3])
    embed = PatchEmbed1(128)
    x = torch_rand(10, 3, 96, 96)

    y = embed(x)

    print("Embed output shape:", y.shape)

    transformer = TransformerLayer(128, 4, False)

    z = transformer(y)

    print("Transformer output shape:", z.shape)





if __name__ == "__main__":

    print("vit.py: main()")

    # obs_shape = [128, 128, 6]  # NHWC
    obs_shape = [6, 128, 128]  # NHWC

    enc = VitEncoder(
        obs_shape,
        VitEncoderConfig(),
        num_channel=obs_shape[-1],
        img_h=obs_shape[0],
        img_w=obs_shape[1],
    )

    print(enc)
    # x = tf.random.uniform([1, *obs_shape]) * 255
    x = torch_rand([1, *obs_shape]) * 255

    print("output size:", enc(x, flatten=False).shape)
    print("repr dim:", enc.repr_dim, ", real dim:", enc(x, flatten=True).shape)

















