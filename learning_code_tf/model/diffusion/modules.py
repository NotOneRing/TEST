import tensorflow as tf
import numpy as np
import math

from util.torch_to_tf import torch_exp, torch_arange, torch_cat, torch_unsqueeze, \
    torch_squeeze

from util.torch_to_tf import nn_Identity, nn_Mish, nn_ReLU, nn_Sequential, nn_GroupNorm, nn_Conv1d, \
nn_ConvTranspose1d


from tensorflow.keras.saving import register_keras_serializable



@register_keras_serializable(package="Custom")
class SinusoidalPosEmb(tf.keras.layers.Layer):
    def __init__(self, dim, name = "SinusoidalPosEmb", **kwargs):

        print("modules.py: SinusoidalPosEmb.__init__()")

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

        # print("modules.py: SinusoidalPosEmb.call()")


        half_dim = int(self.dim // 2)


        emb = math.log(10000) / (half_dim - 1)
        
        # emb = tf.constant(math.log(10000) / (half_dim - 1), dtype=tf.float32)

        # print("1emb.shape = ", emb.shape)
        # print("type(emb) = ", type(emb))
        # print("number emb = ", emb)
        # print("half_dim = ", half_dim)
        
        first_var = torch_arange(start = 0, end = half_dim, step=1)

        first_var = tf.cast(first_var, tf.float32)

        # print("type(first_var) = ", type(first_var))
        # print("first_var = ", first_var)

        second_var = first_var * -emb

        # print("second_var = ", second_var)

        emb = torch_exp( second_var )

        # print("SinusoidalPosEmb emb1 = ", emb)

        # emb = tf.convert_to_tensor(emb, dtype=tf.float32)

        # print("2emb.shape = ", emb.shape)
        # print("type(emb) = ", type(emb))
        # print("emb = ", emb)

        # print("x = ", x)
        
        # x_float32 = tf.cast(x, tf.float32)


        # print("x_float32[:, None] = ", x_float32[:, None])

        # print("emb[None, :] = ", emb[None, :])

        
        emb = tf.cast(x[:, None], tf.float32) * emb[None, :]
        # emb = x[:, None].float() * emb[None, :]

        # print("3emb.shape = ", emb.shape)
        # print("type(emb) = ", type(emb))
        # print("emb = ", emb)

        # print("SinusoidalPosEmb emb2 = ", emb)

        # print("sin(emb).shape = ", tf.sin(emb).shape)
        # print("cos(emb).shape = ", tf.cos(emb).shape)

        emb = torch_cat([tf.sin(emb), tf.cos(emb)], dim=-1)

        # print("4emb.shape = ", emb.shape)
        # print("SinusoidalPosEmb emb = ", emb)

        return emb



class Downsample1d(tf.keras.layers.Layer):
    def __init__(self, dim):

        print("modules.py: Downsample1d.__init__()")

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
    def __init__(self, dim):

        print("modules.py: Upsample1d.__init__()")

        super(Upsample1d, self).__init__()
        # self.conv = tf.keras.layers.Conv1DTranspose(dim, 4, strides=2, padding="same")
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
    ):

        print("modules.py: Conv1dBlock.__init__()")

        super(Conv1dBlock, self).__init__()

        # Mish activation function implementation in TensorFlow
        if activation_type == "Mish":
            act = nn_Mish()
        elif activation_type == "ReLU":
            act = nn_ReLU()
        else:
            raise ValueError("Unknown activation type for Conv1dBlock")


        self.block = nn_Sequential(
            nn_Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            # (lambda x: x.unsqueeze(2)) if n_groups is not None else nn_Identity(),
            (lambda x: torch_unsqueeze(x, 2)) if n_groups is not None else nn_Identity(),
            (
                nn_GroupNorm(n_groups, out_channels, eps=eps)
                if n_groups is not None
                else nn_Identity()
            ),
            # (lambda x: x.squeeze(2)) if n_groups is not None else nn_Identity(),
            (lambda x: torch_squeeze(x, 2)) if n_groups is not None else nn_Identity(),
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


