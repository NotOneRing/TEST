import tensorflow as tf
import numpy as np
import math

from util.torch_to_tf import torch_exp, torch_arange, torch_cat, torch_unsqueeze, \
    torch_squeeze

from util.torch_to_tf import nn_Identity, nn_Mish, nn_ReLU, nn_Sequential, nn_GroupNorm, nn_Conv1d, \
nn_ConvTranspose1d, einops_layers_torch_Rearrange


from tensorflow.keras.saving import register_keras_serializable




@register_keras_serializable(package="Custom")
class SinusoidalPosEmb(tf.keras.layers.Layer):
    def __init__(self, dim, name = "SinusoidalPosEmb", **kwargs):

        super(SinusoidalPosEmb, self).__init__(name=name,**kwargs)
        self.dim = dim

    def get_config(self):
        """Returns the config of the layer for serialization."""
        config = super(SinusoidalPosEmb, self).get_config()
        config.update({"dim": self.dim})
        return config

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        return cls(**config)

    def call(self, x):


        half_dim = int(self.dim // 2)


        emb = math.log(10000) / (half_dim - 1)
        
        
        first_var = torch_arange(start = 0, end = half_dim, step=1)

        first_var = tf.cast(first_var, tf.float32)


        second_var = first_var * -emb


        emb = torch_exp( second_var )


        emb = tf.cast(x[:, None], tf.float32) * emb[None, :]

        emb = torch_cat([tf.sin(emb), tf.cos(emb)], dim=-1)


        return emb



class Downsample1d(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):

        print("modules.py: Downsample1d.__init__()")

        self.dim = dim

        super(Downsample1d, self).__init__()
        self.conv = nn_Conv1d(dim, dim, 3, 2, 1)

    def call(self, x):

        print("modules.py: Downsample1d.call()")

        return self.conv(x)

    def get_config(self):
        config = super(Downsample1d, self).get_config()
        config.update({"dim": self.dim})
        return config
    

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        return cls(**config)    


class Upsample1d(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        
        print("modules.py: Upsample1d.__init__()")

        self.dim = dim

        super(Upsample1d, self).__init__()
        self.conv = nn_ConvTranspose1d(dim, dim, 4, 2, 1)

    def call(self, x):

        print("modules.py: Upsample1d.call()")

        return self.conv(x)

    def get_config(self):
        config = super(Upsample1d, self).get_config()
        config.update({"dim": self.dim})
        return config


    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        return cls(**config)
    




class Conv1dBlock(tf.keras.layers.Layer):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(
        self,
        inp_channels,
        out_channels,
        kernel_size,
        n_groups=None,
        activation_type="Mish",
        eps=1e-5,
        **kwargs
    ):

        print("modules.py: Conv1dBlock.__init__()")


        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.activation_type = activation_type
        self.eps = eps


        super(Conv1dBlock, self).__init__()

        # Mish activation function implementation in TensorFlow
        if activation_type == "Mish":
            act = nn_Mish()
        elif activation_type == "ReLU":
            act = nn_ReLU()
        else:
            raise ValueError("Unknown activation type for Conv1dBlock")

        print("n_groups = ", n_groups)
        print("out_channels = ", out_channels)

        print("padding = ", kernel_size // 2)
        
        self.block = nn_Sequential(
            nn_Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            (
                einops_layers_torch_Rearrange("batch channels horizon -> batch channels 1 horizon", name = "Conv1dBlock_Rearrange1")
                if n_groups is not None
                else nn_Identity()
            ),
            (
                nn_GroupNorm(n_groups, out_channels, eps=eps)
                if n_groups is not None
                else nn_Identity()
            ),
            (
                einops_layers_torch_Rearrange("batch channels 1 horizon -> batch channels horizon", name = "Conv1dBlock_Rearrange2")
                if n_groups is not None
                else nn_Identity()
            ),
            act,
        )



    def call(self, x):

        print("modules.py: Conv1dBlock.call()")

        return self.block(x)




    def get_config(self):
        config = super(Conv1dBlock, self).get_config()
        config.update({
            "inp_channels": self.inp_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "n_groups": self.n_groups,
            "activation_type": self.activation_type,
            "eps": self.eps,
        })
        return config
    

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        return cls(**config)


