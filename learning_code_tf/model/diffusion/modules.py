import tensorflow as tf
import numpy as np

class SinusoidalPosEmb(tf.keras.layers.Layer):
    def __init__(self, dim):

        print("modules.py: SinusoidalPosEmb.__init__()")

        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def call(self, x):

        print("modules.py: SinusoidalPosEmb.call()")

        device = x.device  # TensorFlow handles device management automatically
        half_dim = self.dim // 2


        emb = np.log(10000) / (half_dim - 1)

        print("1emb.shape = ", emb.shape)
        print("type(emb) = ", type(emb))
        print("emb = ", emb)

        emb = np.exp(np.arange(half_dim) * -emb)

        emb = tf.convert_to_tensor(emb, dtype=tf.float32)

        print("2emb.shape = ", emb.shape)
        print("type(emb) = ", type(emb))
        print("emb = ", emb)

        print("x = ", x)
        
        x_float32 = tf.cast(x, tf.float32)


        print("x_float32[:, None] = ", x_float32[:, None])

        print("emb[None, :] = ", emb[None, :])

        
        emb = x_float32[:, None] * emb[None, :]
        # emb = x[:, None].float() * emb[None, :]

        print("3emb.shape = ", emb.shape)
        print("type(emb) = ", type(emb))
        print("emb = ", emb)

        print("sin(emb).shape = ", tf.sin(emb).shape)
        print("cos(emb).shape = ", tf.cos(emb).shape)

        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

        print("4emb.shape = ", emb.shape)
        print("emb = ", emb)

        return emb

    def get_config(self):
        """Returns the config of the layer for serialization."""
        config = super(SinusoidalPosEmb, self).get_config()
        config.update({"dim": self.dim})
        return config

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        return cls(**config)




class Downsample1d(tf.keras.layers.Layer):
    def __init__(self, dim):

        print("modules.py: Downsample1d.__init__()")

        super(Downsample1d, self).__init__()
        self.conv = tf.keras.layers.Conv1D(dim, 3, strides=2, padding="same")

    def call(self, x):

        print("modules.py: Downsample1d.call()")

        return self.conv(x)

    def get_config(self):
        config = super(Downsample1d, self).get_config()
        config.update({"dim": self.dim})
        return config
    
    


class Upsample1d(tf.keras.layers.Layer):
    def __init__(self, dim):

        print("modules.py: Upsample1d.__init__()")

        super(Upsample1d, self).__init__()
        self.conv = tf.keras.layers.Conv1DTranspose(dim, 4, strides=2, padding="same")

    def call(self, x):

        print("modules.py: Upsample1d.call()")

        return self.conv(x)

    def get_config(self):
        config = super(Upsample1d, self).get_config()
        config.update({"dim": self.dim})
        return config


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
            self.activation = tf.keras.layers.Lambda(lambda x: x * tf.tanh(tf.nn.softplus(x)))
        elif activation_type == "ReLU":
            self.activation = tf.keras.layers.ReLU()
        else:
            raise ValueError("Unknown activation type for Conv1dBlock")

        self.conv = tf.keras.layers.Conv1D(
            out_channels, kernel_size, padding="same"
        )

        self.group_norm = None
        if n_groups is not None:
            self.group_norm = tf.keras.layers.GroupNormalization(groups=n_groups, epsilon=eps)

    def call(self, x):

        print("modules.py: Conv1dBlock.call()")

        x = self.conv(x)
        if self.group_norm is not None:
            x = self.group_norm(x)
        x = self.activation(x)
        return x

    # 自己实现的
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