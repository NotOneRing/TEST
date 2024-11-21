import tensorflow as tf
import numpy as np

class SinusoidalPosEmb(tf.keras.layers.Layer):
    def __init__(self, dim):

        print("modules.py: SinusoidalPosEmb.__init__()", flush=True)

        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def call(self, x):

        print("modules.py: SinusoidalPosEmb.call()", flush=True)

        device = x.device  # TensorFlow handles device management automatically
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

class Downsample1d(tf.keras.layers.Layer):
    def __init__(self, dim):

        print("modules.py: Downsample1d.__init__()", flush=True)

        super(Downsample1d, self).__init__()
        self.conv = tf.keras.layers.Conv1D(dim, 3, strides=2, padding="same")

    def call(self, x):

        print("modules.py: Downsample1d.call()", flush=True)

        return self.conv(x)

class Upsample1d(tf.keras.layers.Layer):
    def __init__(self, dim):

        print("modules.py: Upsample1d.__init__()", flush=True)

        super(Upsample1d, self).__init__()
        self.conv = tf.keras.layers.Conv1DTranspose(dim, 4, strides=2, padding="same")

    def call(self, x):

        print("modules.py: Upsample1d.call()", flush=True)

        return self.conv(x)

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

        print("modules.py: Conv1dBlock.__init__()", flush=True)

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

        print("modules.py: Conv1dBlock.call()", flush=True)

        x = self.conv(x)
        if self.group_norm is not None:
            x = self.group_norm(x)
        x = self.activation(x)
        return x
