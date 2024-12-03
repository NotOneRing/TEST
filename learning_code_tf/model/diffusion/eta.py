"""
Eta in DDIM.

Can be learned but always fixed to 1 during training and 0 during eval right now.

"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from model.common.mlp import MLP


class EtaFixed(tf.keras.Model):

    def __init__(
        self,
        base_eta=0.5,
        min_eta=0.1,
        max_eta=1.0,
        **kwargs,
    ):

        print("eta.py: EtaFixed.__init__()")

        super().__init__()
        self.min = min_eta
        self.max = max_eta

        # Initialize eta_logit such that eta = base_eta
        self.eta_logit = tf.Variable(
            tf.math.atanh(2 * (base_eta - min_eta) / (max_eta - min_eta) - 1),
            trainable=True,
            dtype=tf.float32,
        )



    def call(self, cond):
        """Match input batch size, but do not depend on input"""

        print("eta.py: EtaFixed.call()")


        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        batch_size = tf.shape(sample_data)[0]

        eta_normalized = tf.tanh(self.eta_logit)
        # Map to [min, max] from [-1, 1]
        eta = 0.5 * (eta_normalized + 1) * (self.max - self.min) + self.min
        return tf.fill([batch_size, 1], eta)



class EtaAction(tf.keras.Model):

    def __init__(
        self,
        action_dim,
        base_eta=0.5,
        min_eta=0.1,
        max_eta=1.0,
        **kwargs,
    ):

        print("eta.py: EtaAction.__init__()")

        super().__init__()
        # initialize such that eta = base_eta

        self.eta_logit = tf.Variable(
            tf.math.atanh(2 * (base_eta - min_eta) / (max_eta - min_eta) - 1),
            trainable=True,
            dtype=tf.float32,
        )

        self.min = min_eta
        self.max = max_eta

    def call(self, cond):
        """Match input batch size, but do not depend on input"""

        print("eta.py: EtaAction.__call__()")

        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        batch_size = tf.shape(sample_data)[0]

        eta_normalized = tf.tanh(self.eta_logit)
        # Map to [min, max] from [-1, 1]
        eta = 0.5 * (eta_normalized + 1) * (self.max - self.min) + self.min
        return tf.tile(eta, [batch_size, 1])


class EtaState(tf.keras.Model):

    def __init__(
        self,
        input_dim,
        mlp_dims,
        activation_type="ReLU",
        out_activation_type="Identity",
        base_eta=0.5,
        min_eta=0.1,
        max_eta=1.0,
        gain=1e-2,
        **kwargs,
    ):

        print("eta.py: EtaState.__init__()")

        super().__init__()
        self.base = base_eta
        self.min_res = min_eta - base_eta
        self.max_res = max_eta - base_eta
        self.mlp_res = MLP(
            [input_dim] + mlp_dims + [1],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
        )

        # # initialize such that mlp(x) = 0
        # for m in self.mlp_res.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_normal_(m.weight, gain=gain)
        #         m.bias.data.fill_(0)

        # Initialize weights of the MLP to ensure mlp(x) = 0
        for layer in self.mlp_res.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                initializer = tf.keras.initializers.GlorotNormal(gain=gain)
                layer.kernel_initializer = initializer
                layer.bias_initializer = tf.keras.initializers.Zeros()


    def call(self, cond):

        print("eta.py: EtaState.__call__()")

        if "rgb" in cond:
            raise NotImplementedError(
                "State-based eta not implemented for image-based training!"
            )

        # Flatten history
        state = cond["state"]

        B = tf.shape(state)[0]
        state = tf.reshape(state, [B, -1])  # Flatten along batch dimension

        # Forward pass
        eta_res = self.mlp_res(state)
        eta_res = tf.tanh(eta_res)  # [-1, 1]
        eta = eta_res + self.base  # [0, 2]
        return tf.clip_by_value(eta, self.min_res + self.base, self.max_res + self.base)


class EtaStateAction(tf.keras.Model):

    def __init__(
        self,
        input_dim,
        mlp_dims,
        action_dim,
        activation_type="ReLU",
        out_activation_type="Identity",
        base_eta=1,
        min_eta=1e-3,
        max_eta=2,
        gain=1e-2,
        **kwargs,
    ):

        print("eta.py: EtaStateAction.__init__()")

        super().__init__()
        self.base = base_eta
        self.min_res = min_eta - base_eta
        self.max_res = max_eta - base_eta
        self.mlp_res = MLP(
            [input_dim] + mlp_dims + [action_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
        )


        # Initialize weights of the MLP to ensure mlp(x) = 0
        for layer in self.mlp_res.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                initializer = tf.keras.initializers.GlorotNormal(gain=gain)
                layer.kernel_initializer = initializer
                layer.bias_initializer = tf.keras.initializers.Zeros()


    def call(self, cond):

        print("eta.py: EtaStateAction.__call__()")

        if "rgb" in cond:
            raise NotImplementedError(
                "State-action-based eta not implemented for image-based training!"
            )

        # Flatten history
        state = cond["state"]
        B = tf.shape(state)[0]
        state = tf.reshape(state, [B, -1])  # Flatten along batch dimension

        # Forward pass
        eta_res = self.mlp_res(state)
        eta_res = tf.tanh(eta_res)  # [-1, 1]
        eta = eta_res + self.base
        return tf.clip_by_value(eta, self.min_res + self.base, self.max_res + self.base)










