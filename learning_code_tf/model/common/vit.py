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
        cfg_embed_style = None,
        cfg_embed_dim = None,
        cfg_embed_norm = None,
        cfg_num_heads = None,
        cfg_depth = None,
        **kwargs
    ):

        print("vit.py: VitEncoder.__init__()")

        super().__init__()
        
        self.obs_shape = list(obs_shape)

        self.cfg = cfg
        self.num_channel = num_channel


        if cfg:
            self.cfg_embed_style = cfg.embed_style
            self.cfg_embed_dim = cfg.embed_dim
            self.cfg_embed_norm = cfg.embed_norm
            self.cfg_num_heads = cfg.num_heads
            self.cfg_depth = cfg.depth

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
        else:
            self.cfg_embed_style = cfg_embed_style
            self.cfg_embed_dim = cfg_embed_dim
            self.cfg_embed_norm = cfg_embed_norm
            self.cfg_num_heads = cfg_num_heads
            self.cfg_depth = cfg_depth

            self.vit = MinVit(
                embed_style=cfg_embed_style,
                embed_dim=cfg_embed_dim,
                embed_norm=cfg_embed_norm,
                num_head=cfg_num_heads,
                depth=cfg_depth,
                num_channel=num_channel,
                img_h=img_h,
                img_w=img_w,
            )

        self.img_h = img_h
        self.img_w = img_w
        self.num_patch = self.vit.num_patches
        if self.cfg:
            self.patch_repr_dim = self.cfg.embed_dim
            self.repr_dim = self.cfg.embed_dim * self.vit.num_patches
        else:
            self.patch_repr_dim = cfg_embed_dim
            self.repr_dim = self.cfg_embed_dim * self.vit.num_patches

    def get_config(self):
        """Returns the config of the layer for serialization."""
        config = super(VitEncoder, self).get_config()
        
        print(f"obs_shape: {self.obs_shape}, type: {type(self.obs_shape)}")
        print(f"cfg: {self.cfg}, type: {type(self.cfg)}")
        print(f"num_channel: {self.num_channel}, type: {type(self.num_channel)}")
        print(f"img_h: {self.img_h}, type: {type(self.img_h)}")
        print(f"img_w: {self.img_w}, type: {type(self.img_w)}")


        config.update({
                        "obs_shape": self.obs_shape,
                        "cfg": None,
                        "num_channel": self.num_channel,
                        "img_h": self.img_h,
                        "img_w": self.img_w,
                        "cfg_embed_style" : self.cfg_embed_style,
                        "cfg_embed_dim" : self.cfg_embed_dim,
                        "cfg_embed_norm" : self.cfg_embed_norm,
                        "cfg_num_heads" : self.cfg_num_heads,
                        "cfg_depth" : self.cfg_depth
                       })

        config.update({
            "vit": tf.keras.layers.serialize(self.vit),
        })
        
        return config

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        
        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'MinVit': MinVit, 
         }
        # Register your custom class with Keras
        get_custom_objects().update(cur_dict)

        
        vit = tf.keras.layers.deserialize( config.pop("vit"),  custom_objects=get_custom_objects() )
        
        
        return cls(vit=vit, **config)


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
    def __init__(self, embed_dim, num_channel=3, img_h=96, img_w=96, conv = None,
        **kwargs):

        print("vit.py: PatchEmbed1.__init__()")

        super().__init__()

        self.embed_dim = embed_dim
        self.num_channel = num_channel
        self.img_h = img_h
        self.img_w = img_w
        if conv:
            self.conv = conv
        else:
            self.conv = nn_Conv2d(num_channel, embed_dim, kernel_size=8, stride=8)
        self.num_patch = math.ceil(img_h / 8) * math.ceil(img_w / 8)
        self.patch_dim = embed_dim

    def get_config(self):
        """Returns the config of the layer for serialization."""
        config = super(PatchEmbed1, self).get_config()

        print(f"embed_dim: {self.embed_dim}, type: {type(self.embed_dim)}")
        print(f"num_channel: {self.num_channel}, type: {type(self.num_channel)}")
        print(f"img_h: {self.img_h}, type: {type(self.img_h)}")
        print(f"img_w: {self.img_w}, type: {type(self.img_w)}")

        config.update({
                        "embed_dim": self.embed_dim,
                        "num_channel": self.num_channel,
                        "img_h": self.img_h,
                        "img_w": self.img_w,
                       })

        config.update({
            "conv": tf.keras.layers.serialize(self.conv),
        })
        
        return config

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        
        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'nn_Conv2d': nn_Conv2d,
            'nn_GroupNorm': nn_GroupNorm,
            'nn_ReLU': nn_ReLU,
            'nn_Sequential': nn_Sequential,
            'nn_Linear': nn_Linear,
         }

        get_custom_objects().update(cur_dict)

        conv = tf.keras.layers.deserialize( config.pop("conv"),  custom_objects=get_custom_objects() )

        return cls(conv=conv, **config)


    def call(self, x):

        print("vit.py: PatchEmbed1.call()")

        y = self.conv(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c")

        return y


class PatchEmbed2(tf.keras.layers.Layer):
    def __init__(self, embed_dim, use_norm, num_channel=3, img_h=96, img_w=96, embed = None,
        **kwargs):

        print("vit.py: PatchEmbed2.__init__()")

        super().__init__()

        self.embed_dim = embed_dim
        self.use_norm = use_norm
        self.num_channel = num_channel
        self.img_h = img_h
        self.img_w = img_w

        #输入是num_channel
        layers = [
            nn_Conv2d(num_channel, embed_dim, kernel_size=8, stride=4),
            nn_GroupNorm() if use_norm else nn_Identity(),
            nn_ReLU(),
            nn_Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2),
        ]

        if embed:
            self.embed = embed
        else:
            self.embed = nn_Sequential(layers)

        H1 = math.ceil((img_h - 8) / 4) + 1
        W1 = math.ceil((img_w - 8) / 4) + 1
        H2 = math.ceil((H1 - 3) / 2) + 1
        W2 = math.ceil((W1 - 3) / 2) + 1
        self.num_patch = H2 * W2
        self.patch_dim = embed_dim

    def get_config(self):
        """Returns the config of the layer for serialization."""
        config = super(PatchEmbed2, self).get_config()

        print(f"embed_dim: {self.embed_dim}, type: {type(self.embed_dim)}")
        print(f"use_norm: {self.use_norm}, type: {type(self.use_norm)}")
        print(f"num_channel: {self.num_channel}, type: {type(self.num_channel)}")
        print(f"img_h: {self.img_h}, type: {type(self.img_h)}")
        print(f"img_w: {self.img_w}, type: {type(self.img_w)}")

        config.update({
                        "embed_dim": self.embed_dim,
                        "use_norm": self.use_norm,
                        "num_channel": self.num_channel,
                        "img_h": self.img_h,
                        "img_w": self.img_w,
                       })

        config.update({
            "embed": tf.keras.layers.serialize(self.embed),
        })
        
        return config

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        
        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'nn_Conv2d': nn_Conv2d,
            'nn_GroupNorm': nn_GroupNorm,
            'nn_ReLU': nn_ReLU,
            'nn_Sequential': nn_Sequential,
            'nn_Linear': nn_Linear,
         }

        get_custom_objects().update(cur_dict)

        embed = tf.keras.layers.deserialize( config.pop("embed"),  custom_objects=get_custom_objects() )

        return cls(embed=embed, **config)


    def call(self, x):

        print("vit.py: PatchEmbed2.call()")

        y = self.embed(x)
        y = einops.rearrange(y, "b c h w -> b (h  w) c")
        return y


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_head, qkv_proj = None, out_proj = None,
        **kwargs):

        print("vit.py: MultiHeadAttention.__init__()")

        super().__init__()
        assert embed_dim % num_head == 0
        self.embed_dim = embed_dim
        self.num_head = num_head

        if qkv_proj:
            self.qkv_proj = qkv_proj
        else:
            self.qkv_proj = nn_Linear(embed_dim, 3 * embed_dim)

        if out_proj:
            self.out_proj = out_proj
        else:
            self.out_proj = nn_Linear(embed_dim, embed_dim)

    def get_config(self):
        """Returns the config of the layer for serialization."""
        config = super(MultiHeadAttention, self).get_config()

        print(f"embed_dim: {self.embed_dim}, type: {type(self.embed_dim)}")
        print(f"num_head: {self.num_head}, type: {type(self.num_head)}")

        config.update({
                        "embed_dim": self.embed_dim,
                        "num_head": self.num_head,
                       })

        config.update({
            "qkv_proj": tf.keras.layers.serialize(self.qkv_proj),
            "out_proj": tf.keras.layers.serialize(self.out_proj),
        })
        
        return config

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        
        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'nn_Linear': nn_Linear,
         }

        get_custom_objects().update(cur_dict)

        qkv_proj = tf.keras.layers.deserialize( config.pop("qkv_proj"),  custom_objects=get_custom_objects() )
        out_proj = tf.keras.layers.deserialize( config.pop("out_proj"),  custom_objects=get_custom_objects() )

        return cls(qkv_proj=qkv_proj, out_proj = out_proj, **config)


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
    def __init__(self, embed_dim, num_head, dropout,
                    layer_norm1=None, mha = None, layer_norm2 = None, 
                    linear1=None, linear2 = None, dropout_layer = None,
                    **kwargs                  
                ):

        print("vit.py: TransformerLayer.__init__()")

        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.initial_dropout = dropout

        if layer_norm1:
            self.layer_norm1 = layer_norm1
        else:
            self.layer_norm1 = nn_LayerNorm(embed_dim)
        if mha:
            self.mha = mha
        else:
            self.mha = MultiHeadAttention(embed_dim, num_head)
        if layer_norm2:
            self.layer_norm2 = layer_norm2
        else:
            self.layer_norm2 = nn_LayerNorm(embed_dim)
        if linear1:
            self.linear1 = linear1
        else:
            self.linear1 = nn_Linear(embed_dim, 4 * embed_dim)
        if linear2:
            self.linear2 = linear2
        else:
            self.linear2 = nn_Linear(4 * embed_dim, embed_dim)
        if dropout_layer:
            self.dropout = dropout_layer
        else:
            self.dropout = nn_Dropout(dropout)

    def get_config(self):
        """Returns the config of the layer for serialization."""
        config = super(TransformerLayer, self).get_config()

        print(f"embed_dim: {self.embed_dim}, type: {type(self.embed_dim)}")
        print(f"num_head: {self.num_head}, type: {type(self.num_head)}")
        print(f"dropout: {self.dropout}, type: {type(self.dropout)}")
        config.update({
                        "embed_dim": self.embed_dim,
                        "num_head": self.num_head,
                        "dropout": self.initial_dropout
                       })
        config.update({
            "layer_norm1": tf.keras.layers.serialize(self.layer_norm1),
            "mha": tf.keras.layers.serialize(self.mha),
            "layer_norm2": tf.keras.layers.serialize(self.layer_norm2),
            "linear1": tf.keras.layers.serialize(self.linear1),
            "linear2": tf.keras.layers.serialize(self.linear2),
            "dropout_layer": tf.keras.layers.serialize(self.dropout),
        })
        
        return config

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        
        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'nn_Sequential': nn_Sequential, 
            'nn_LayerNorm': nn_LayerNorm,
            'MultiHeadAttention': MultiHeadAttention, 
            'nn_Linear': nn_Linear,
            "nn_Dropout": nn_Dropout
         }
        # Register your custom class with Keras
        get_custom_objects().update(cur_dict)

        
        layer_norm1 = tf.keras.layers.deserialize( config.pop("layer_norm1"),  custom_objects=get_custom_objects() )
        mha = tf.keras.layers.deserialize( config.pop("mha"),  custom_objects=get_custom_objects() )
        layer_norm2 = tf.keras.layers.deserialize( config.pop("layer_norm2"),  custom_objects=get_custom_objects() )

        linear1 = tf.keras.layers.deserialize( config.pop("linear1"),  custom_objects=get_custom_objects() )
        linear2 = tf.keras.layers.deserialize( config.pop("linear2"),  custom_objects=get_custom_objects() )
        dropout_layer = tf.keras.layers.deserialize( config.pop("dropout_layer"),  custom_objects=get_custom_objects() )


        return cls(layer_norm1=layer_norm1, mha = mha, layer_norm2 = layer_norm2, 
                   linear1=linear1, linear2 = linear2, dropout_layer = dropout_layer, 
                   **config)


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
        patch_embed = None,
        net = None,
        norm = None,
        **kwargs
    ):

        print("vit.py: MinVit.__init__()")

        super().__init__()
        
        self.embed_style = embed_style
        self.embed_dim = embed_dim
        self.embed_norm = embed_norm
        self.num_head = num_head
        self.depth = depth
        self.num_channel = num_channel
        self.img_h = img_h
        self.img_w = img_w

        if embed_style == "embed1":
            if patch_embed:
                self.patch_embed = patch_embed
            else:
                self.patch_embed = PatchEmbed1(
                    embed_dim,
                    num_channel=num_channel,
                    img_h=img_h,
                    img_w=img_w,
                )
        elif embed_style == "embed2":
            if patch_embed:
                self.patch_embed = patch_embed
            else:
                self.patch_embed = PatchEmbed2(
                    embed_dim,
                    use_norm=embed_norm,
                    num_channel=num_channel,
                    img_h=img_h,
                    img_w=img_w,
                )
        else:
            assert False

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


    def get_config(self):
        """Returns the config of the layer for serialization."""
        config = super(MinVit, self).get_config()

        print(f"embed_style: {self.embed_style}, type: {type(self.embed_style)}")
        print(f"embed_dim: {self.embed_dim}, type: {type(self.embed_dim)}")
        print(f"embed_norm: {self.embed_norm}, type: {type(self.embed_norm)}")
        print(f"num_head: {self.num_head}, type: {type(self.num_head)}")
        print(f"depth: {self.depth}, type: {type(self.depth)}")

        print(f"num_channel: {self.num_channel}, type: {type(self.num_channel)}")
        print(f"img_h: {self.img_h}, type: {type(self.img_h)}")
        print(f"img_w: {self.img_w}, type: {type(self.img_w)}")


        config.update({
                        "embed_style": self.embed_style,
                        "embed_dim": self.embed_dim,
                        "embed_norm": self.embed_norm,
                        "num_head": self.num_head,
                        "depth" : self.depth,
                        "num_channel" : self.num_channel,
                        "img_h" : self.img_h,
                        "img_w" : self.img_w,
                       })


        config.update({
            "patch_embed": tf.keras.layers.serialize(self.patch_embed),
            "net": tf.keras.layers.serialize(self.net),
            "norm": tf.keras.layers.serialize(self.norm),
        })
        
        return config

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        
        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'nn_Sequential': nn_Sequential, 
            'nn_LayerNorm': nn_LayerNorm,
            "TransformerLayer": TransformerLayer
         }
        # Register your custom class with Keras
        get_custom_objects().update(cur_dict)

        
        patch_embed = tf.keras.layers.deserialize( config.pop("patch_embed"),  custom_objects=get_custom_objects() )
        net = tf.keras.layers.deserialize( config.pop("net"),  custom_objects=get_custom_objects() )
        norm = tf.keras.layers.deserialize( config.pop("norm"),  custom_objects=get_custom_objects() )
        

        return cls(patch_embed=patch_embed, net = net, norm = norm, **config)


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

















