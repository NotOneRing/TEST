

import tensorflow as tf
# from tensorflow.keras import layers, models
from tensorflow.keras import models

from collections import OrderedDict
import logging

from util.torch_to_tf import nn_ReLU, nn_GELU, nn_Tanh, nn_ELU, nn_Mish, nn_Identity, nn_Softplus

# activation_dict = {
#     "ReLU": tf.keras.layers.ReLU(),
#     "ELU": tf.keras.layers.ELU(),
#     "GELU": tf.keras.layers.Activation(tf.keras.activations.gelu),  # 使用 Activation 层来包装 GELU 函数
#     "Tanh": tf.keras.layers.Activation(tf.keras.activations.tanh),  # 使用 tf.keras.activations.tanh
#     "Mish": tf.keras.layers.Activation(lambda x: x * tf.tanh(tf.math.log(1 + tf.exp(x)))),  # Custom Mish implementation
#     "Identity": tf.keras.layers.Activation("linear"),
#     "Softplus": tf.keras.layers.Activation(tf.keras.activations.softplus),  # 使用 tf.keras.activations.softplus
# }


activation_dict = {
    "ReLU": nn_ReLU(),
    "ELU": nn_ELU(),
    "GELU": nn_GELU(),
    "Tanh": nn_Tanh(),
    "Mish": nn_Mish(),
    "Identity": nn_Identity(),
    "Softplus": nn_Softplus(),
}


class MLP(models.Model):
    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="Tanh",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        use_drop_final=False,
        verbose=False,
    ):
        print("mlp.py: MLP.__init__()")

        super(MLP, self).__init__()

        self.append_layers = append_layers
        num_layer = len(dim_list) - 1
        self.moduleList = []

        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in append_layers:
                i_dim += append_dim
            # layers_list = [("linear_1", tf.keras.layers.Dense(o_dim))]
            layers_list = [tf.keras.layers.Dense(o_dim)]
            
            # Add normalization and dropout
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                # layers_list.append(("norm_1", tf.keras.layers.LayerNormalization()))
                layers_list.append( tf.keras.layers.LayerNormalization() )
            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                # layers_list.append(("dropout_1", tf.keras.layers.Dropout(dropout)))
                layers_list.append( tf.keras.layers.Dropout(dropout) )
            
            # Add activation function
            act = (
                activation_dict[activation_type]
                if idx != num_layer - 1
                else activation_dict[out_activation_type]
            )
            # layers_list.append(("act_1", act))
            layers_list.append(act)

            print('before self.moduleList.append')

            # temp = OrderedDict(layers_list)

            # print("temp = ", temp)

            # Append to model layers
            self.moduleList.append(tf.keras.Sequential(layers_list))

            # layers = []
            # for layer in layers_list:
            #     layers.append()

        if verbose:
            logging.info(self.moduleList)


    def get_config(self):
        config = super(MLP, self).get_config()  # Call the base class's get_config
        config.update({
            "dim_list": self.dim_list,
            "append_dim": self.append_dim,
            "append_layers": self.append_layers,
            "activation_type": self.activation_type,
            "out_activation_type": self.out_activation_type,
            "use_layernorm": self.use_layernorm,
            "use_layernorm_final": self.use_layernorm_final,
            "dropout": self.dropout,
            "use_drop_final": self.use_drop_final,
        })
        return config
    

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        return cls(**config)


    
    def call(self, x, append=None):
        print("mlp.py: MLP.call()")

        for layer_ind, m in enumerate(self.moduleList):
            if append is not None and layer_ind in self.append_layers:
                x = tf.concat([x, append], axis=-1)
            x = m(x)
        return x



class ResidualMLP(models.Model):
    def __init__(
        self,
        dim_list,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
    ):
        print("mlp.py: ResidualMLP.__init__()")

        super(ResidualMLP, self).__init__()

        print("after super()")

        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert num_hidden_layers % 2 == 0

        print("after dim")

        # self.cur_layers = [tf.keras.layers.Dense(hidden_dim)]

        # print("after layers")

        # self.cur_layers.extend(
        #     [
        #         TwoLayerPreActivationResNetLinear(
        #             hidden_dim=hidden_dim,
        #             activation_type=activation_type,
        #             use_layernorm=use_layernorm,
        #             dropout=dropout,
        #         )
        #         for _ in range(1, num_hidden_layers, 2)
        #     ]
        # )

        # print("after layers.extend()")

        # self.cur_layers.append(tf.keras.layers.Dense(dim_list[-1]))

        # print("after append()")

        # if use_layernorm_final:
        #     self.cur_layers.append(tf.keras.layers.LayerNormalization())

        # print("after use_layernorm_final")

        # self.cur_layers.append(activation_dict[out_activation_type])


        self.cur_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim),
            *[
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
                for _ in range(1, num_hidden_layers, 2)
            ],
            tf.keras.layers.Dense(dim_list[-1]),
        ])

        if use_layernorm_final:
            self.cur_layers.add(tf.keras.layers.LayerNormalization())

        self.cur_layers.add(activation_dict[out_activation_type])




        # print("after append()")


    def get_config(self):
        config = super(ResidualMLP, self).get_config()  # Call the base class's get_config
        config.update({
            "dim_list": self.cur_layers[0].input_shape[1:],  # Store dim_list (using input_shape to capture dims)
            "activation_type": self.activation_type,
            "out_activation_type": self.out_activation_type,
            "use_layernorm": self.use_layernorm,
            "use_layernorm_final": self.use_layernorm_final,
            "dropout": self.dropout,
        })
        return config
    

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        return cls(**config)

    

    def call(self, x):
        print("mlp.py: ResidualMLP.call()")

        # for cur_layers in self.cur_layers:
        #     x = cur_layers(x)
        # x = self.cur_layers(x)
        # return x
        return self.cur_layers(x)
    




class TwoLayerPreActivationResNetLinear(models.Model):
    def __init__(
        self,
        hidden_dim,
        activation_type="Mish",
        use_layernorm=False,
        dropout=0,
    ):
        print("mlp.py: TwoLayerPreActivationResNetLinear.__init__()")

        super().__init__()
        self.l1 = tf.keras.layers.Dense(hidden_dim)
        self.l2 = tf.keras.layers.Dense(hidden_dim)
        self.act = activation_dict[activation_type]
        if use_layernorm:
            self.norm1 = tf.keras.layers.LayerNormalization()
            self.norm2 = tf.keras.layers.LayerNormalization()

        if dropout > 0:
            raise NotImplementedError("Dropout not implemented for residual MLP!")

    def call(self, x):
        print("mlp.py: TwoLayerPreActivationResNetLinear.call()")

        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))
        result = x + x_input
        return result

