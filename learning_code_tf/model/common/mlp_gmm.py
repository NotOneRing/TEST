"""
MLP models for GMM policy.

"""

import tensorflow as tf

from model.common.mlp import MLP, ResidualMLP

from util.torch_to_tf import nn_Parameter, torch_log, torch_tensor

class GMM_MLP(tf.keras.Model):

    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim=None,
        mlp_dims=[256, 256, 256],
        num_modes=5,
        activation_type="Mish",
        residual_style=False,
        use_layernorm=False,
        fixed_std=None,
        learn_fixed_std=False,
        std_min=0.01,
        std_max=1,
    ):

        print("mlp_gmm.py: GMM_MLP.__init__()")

        super().__init__()
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        input_dim = cond_dim
        output_dim = action_dim * horizon_steps * num_modes
        self.num_modes = num_modes
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
            self.mlp_logvar = model(
                [input_dim] + mlp_dims + [output_dim],
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )
        elif (
            learn_fixed_std
        ):  # initialize to fixed_std, separate for each action and mode            
            # Learnable fixed_std
            # self.logvar = self.add_weight(
            #     shape=(action_dim * num_modes,),
            #     initializer=tf.constant_initializer(
            #         tf.math.log([fixed_std**2] * (action_dim * num_modes))
            #     ),
            #     trainable=True,
            #     name="logvar",
            # )
            self.logvar = nn_Parameter(
                torch_log(
                    torch_tensor(
                        [fixed_std**2 for _ in range(action_dim * num_modes)]
                    )
                ),
                requires_grad=True,
            )


        # self.logvar_min = tf.constant(
        #     tf.math.log(std_min**2), dtype=tf.float32, name="logvar_min"
        # )
        # self.logvar_max = tf.constant(
        #     tf.math.log(std_max**2), dtype=tf.float32, name="logvar_max"
        # )
        self.logvar_min = nn_Parameter(
            torch_log(torch_tensor(std_min**2)), requires_grad=False
        )
        self.logvar_max = nn_Parameter(
            torch_log(torch_tensor(std_max**2)), requires_grad=False
        )


        self.use_fixed_std = fixed_std is not None
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std

        # mode weights
        self.mlp_weights = model(
            [input_dim] + mlp_dims + [num_modes],
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )


    def call(self, cond):

        print("mlp_gmm.py: GMM_MLP.call()")

        B = len(cond["state"])
        # device = cond["state"].device

        # flatten history
        state = torch_tensor_view(cond["state"], [B, -1])

        # mlp
        out_mean = self.mlp_mean(state)

        out_mean = torch_tanh(out_mean)
        out_mean = torch_tensor_view(
            out_mean, [B, self.num_modes, self.horizon_steps * self.action_dim]
        ) # tanh squashing in [-1, 1]


        if self.learn_fixed_std:
            out_logvar = torch_clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch_exp(0.5 * out_logvar)
            out_scale = torch_tensor_view(
                out_scale, [1, self.num_modes, self.action_dim]
            )
            out_scale = torch_tensor_repeat(out_scale, [B, 1, self.horizon_steps])

        elif self.use_fixed_std:
            out_scale = torch_ones_like(out_mean) * self.fixed_std
        else:

            out_logvar = self.mlp_logvar(state)
            out_logvar = torch_tensor_view( out_logvar,
                B, self.num_modes, self.horizon_steps * self.action_dim
            )

            out_logvar = torch_clamp(out_logvar, self.logvar_min, self.logvar_max)
            out_scale = torch_exp(0.5 * out_logvar)

        out_weights = self.mlp_weights(state)

        out_weights = torch_tensor_view(out_weights, [B, self.num_modes])

        return out_mean, out_scale, out_weights











