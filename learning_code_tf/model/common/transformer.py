"""
Implementation of Transformer, parameterized as Gaussian and GMM.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py

"""

import logging

import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np

from model.diffusion.modules import SinusoidalPosEmb

logger = logging.getLogger(__name__)


from util.torch_to_tf import torch_zeros, torch_unsqueeze, torch_ones

# class Gaussian_Transformer(nn.Module):
class Gaussian_Transformer(tf.keras.Model):
    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        transformer_embed_dim=256,
        transformer_num_heads=8,
        transformer_num_layers=6,
        transformer_activation="gelu",
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        fixed_std=None,
        learn_fixed_std=False,
        std_min=0.01,
        std_max=1,
    ):
        
        print("transformer.py: Gaussian_Transformer.__init__()")

        super().__init__()
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        output_dim = action_dim

        if fixed_std is None:  # learn the logvar
            output_dim *= 2  # mean and logvar
            logger.info("Using learned std")
        elif learn_fixed_std:  # learn logvar
            self.logvar = tf.Variable(
                initial_value=np.log(fixed_std ** 2) * np.ones(action_dim, dtype=np.float32),
                trainable=True,
            )
            logger.info(f"Using fixed std {fixed_std} with learning")
        else:
            logger.info(f"Using fixed std {fixed_std} without learning")


        self.logvar_min = tf.Variable(np.log(std_min ** 2), trainable=False)
        self.logvar_max = tf.Variable(np.log(std_max ** 2), trainable=False)


        self.learn_fixed_std = learn_fixed_std
        self.fixed_std = fixed_std

        self.transformer = Transformer(
            output_dim,
            horizon_steps,
            cond_dim,
            T_cond=1,  # right now we assume only one step of observation everywhere
            n_layer=transformer_num_layers,
            n_head=transformer_num_heads,
            n_emb=transformer_embed_dim,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            activation=transformer_activation,
        )

    def call(self, cond):

        print("transformer.py: Gaussian_Transformer.forward()")

        B = len(cond["state"])

        # flatten history
        state = tf.reshape(cond["state"], (B, -1))

        # input to transformer
        state = tf.expand_dims(state, axis=1)  # (B,1,cond_dim)
        out, _ = self.transformer(state)  # (B,horizon,output_dim)

        # # use the first half of the output as mean
        assert self.num_modes == 1, "self.num_modes != 1, the code is not congruent with the PyTorch Version!"
        out_mean = tf.tanh(out[:, :, : self.action_dim])
        out_mean = tf.reshape(out_mean, (B, self.horizon_steps * self.action_dim))



        if self.learn_fixed_std:
            out_logvar = tf.clip_by_value(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = tf.exp(0.5 * out_logvar)
            out_scale = tf.reshape(out_scale, (1, self.action_dim))
            out_scale = tf.tile(out_scale, [B, self.horizon_steps])  

        elif self.fixed_std is not None:
            out_scale = tf.ones_like(out_mean) * self.fixed_std
        else:
            out_logvar = out[:, :, self.action_dim:]
            out_logvar = tf.reshape(out_logvar, (B, self.horizon_steps * self.action_dim))
            out_logvar = tf.clip_by_value(out_logvar, self.logvar_min, self.logvar_max)
            out_scale = tf.exp(0.5 * out_logvar)

        return out_mean, out_scale


# class GMM_Transformer(nn.Module):
class GMM_Transformer(tf.keras.Model):
    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        num_modes=5,
        transformer_embed_dim=256,
        transformer_num_heads=8,
        transformer_num_layers=6,
        transformer_activation="gelu",
        p_drop_emb=0,
        p_drop_attn=0,
        fixed_std=None,
        learn_fixed_std=False,
        std_min=0.01,
        std_max=1,
    ):

        print("transformer.py: GMM_Transformer.__init__()")

        super().__init__()
        self.num_modes = num_modes
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        output_dim = action_dim * num_modes

        if fixed_std is None:
            output_dim += num_modes * action_dim  # logvar for each mode
            logger.info("Using learned std")
        elif (
            learn_fixed_std
        ):  # initialize to fixed_std, separate for each action and mode, but same along horizon
            self.logvar = tf.Variable(
                initial_value=np.log(fixed_std ** 2) * np.ones(num_modes * action_dim, dtype=np.float32),
                trainable=True,
            )
            logger.info(f"Using fixed std {fixed_std} with learning")
        else:
            logger.info(f"Using fixed std {fixed_std} without learning")

        self.logvar_min = tf.Variable(np.log(std_min ** 2), trainable=False)
        self.logvar_max = tf.Variable(np.log(std_max ** 2), trainable=False)
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std

        self.transformer = Transformer(
            output_dim,
            horizon_steps,
            cond_dim,
            T_cond=1,  # right now we assume only one step of observation everywhere
            n_layer=transformer_num_layers,
            n_head=transformer_num_heads,
            n_emb=transformer_embed_dim,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            activation=transformer_activation,
        )

        #输入维度horizon_steps * transformer_embed_dim
        self.modes_head = layers.Dense(num_modes)

    def call(self, cond):

        print("transformer.py: GMM_Transformer.call()")

        B = len(cond["state"])

        # flatten history
        state = cond["state"].view(B, -1)

        # input to transformer
        # state = state.unsqueeze(1)  # (B,1,cond_dim)
        state = torch_unsqueeze(state, 1)  # (B,1,cond_dim)

        out, out_prehead = self.transformer(
            state
        )  # (B,horizon,output_dim), (B,horizon,emb_dim)

        # use the first half of the output as mean
        out_mean = tf.tanh(out[:, :, :self.num_modes * self.action_dim])

        out_mean = tf.reshape(out_mean, (B, self.horizon_steps, self.num_modes, self.action_dim))

        out_mean = tf.transpose(out_mean, (0, 2, 1, 3))  # flip horizons and modes

        out_mean = tf.reshape(out_mean, (B, self.num_modes, self.horizon_steps * self.action_dim))

        if self.learn_fixed_std:
            out_logvar = tf.clip_by_value(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = tf.exp(0.5 * out_logvar)
            out_scale = tf.reshape(out_scale, (1, self.num_modes, self.action_dim))
            out_scale = tf.tile(out_scale, [B, 1, self.horizon_steps])

        elif self.fixed_std is not None:
            out_scale = tf.ones_like(out_mean) * self.fixed_std
        else:
            out_logvar = out[
                :, :, self.num_modes * self.action_dim : -self.num_modes
            ]
            out_logvar = out_logvar.reshape(
                B, self.horizon_steps, self.num_modes, self.action_dim
            )

            out_logvar = tf.transpose(out_logvar, (0, 2, 1, 3))  # flip horizons and modes

            out_logvar = tf.reshape(out_logvar, (B, self.num_modes, self.horizon_steps * self.action_dim))

            out_logvar = tf.clip_by_value(out_logvar, self.logvar_min, self.logvar_max)
            out_scale = tf.exp(0.5 * out_logvar)

        # use last horizon step as the mode weights - as it depends on the entire context
        out_weights = self.modes_head(tf.reshape(out_prehead, (B, -1)))

        return out_mean, out_scale, out_weights
















class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_feedforward, activation=activation),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training):
        # Self-attention
        attn_output = self.self_attn(x, x, training=training)
        x = x + self.dropout1(attn_output, training=training)
        x = self.norm1(x)

        # Feedforward network
        ffn_output = self.ffn(x, training=training)
        x = x + self.dropout2(ffn_output, training=training)
        x = self.norm2(x)
        return x


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, nhead, dim_feedforward, dropout, activation):
        super(TransformerEncoder, self).__init__()
        self.layers = [
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(n_layers)
        ]

    def call(self, x, training):
        for layer in self.layers:
            x = layer(x, training=training)
        return x



class Mish(tf.keras.layers.Layer):
    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
        self.cross_attn = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_feedforward, activation=activation),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, tgt, memory, tgt_mask=None, memory_mask=None, training=None):
        # Self-attention on target
        tgt2 = self.self_attn(tgt, tgt, attention_mask=tgt_mask, training=training)
        tgt = tgt + self.dropout1(tgt2, training=training)
        tgt = self.norm1(tgt)

        # Cross-attention between target and memory
        tgt2 = self.cross_attn(tgt, memory, attention_mask=memory_mask, training=training)
        tgt = tgt + self.dropout2(tgt2, training=training)
        tgt = self.norm2(tgt)

        # Feedforward network
        tgt2 = self.ffn(tgt, training=training)
        tgt = tgt + self.dropout3(tgt2, training=training)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, nhead, dim_feedforward, dropout, activation):
        super(TransformerDecoder, self).__init__()
        self.layers = [
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(n_layers)
        ]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, tgt, memory, tgt_mask=None, memory_mask=None, training=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, training=training)
        return self.norm(tgt)

        

class Transformer(tf.keras.Model):
    def __init__(
        self,
        output_dim,
        horizon,
        cond_dim,
        T_cond=1,
        n_layer=12,
        n_head=12,
        n_emb=768,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        causal_attn=False,
        n_cond_layers=0,
        activation="gelu",
    ):

        print("transformer.py: Transformer.__init__()")

        super().__init__()

        # encoder for observations
        #输入维度cond_dim
        self.cond_obs_emb = layers.Dense(n_emb)
        self.cond_pos_emb = tf.Variable(tf.zeros([1, T_cond, n_emb]))


        if n_cond_layers > 0:
            self.encoder = TransformerEncoder(
                n_layers=n_cond_layers,
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation=activation,
            )


        else:
            # 输入n_emb
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(4 * n_emb),
                Mish(),
                tf.keras.layers.Dense(n_emb),
            ])

        # decoder
        self.pos_emb = nn.Parameter(torch_zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation=activation,
            batch_first=True,
            norm_first=True,  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_layer
        )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = horizon
            mask = (torch.triu(torch_ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("mask", mask)

            t, s = torch.meshgrid(
                torch.arange(horizon), torch.arange(T_cond), indexing="ij"
            )
            mask = t >= (
                s - 1
            )  # add one dimension since time is the first token in cond
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("memory_mask", mask)
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # constants
        self.T_cond = T_cond
        self.horizon = horizon

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module):

        print("transformer.py: Transformer._init_weights()")

        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, Transformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(
        self,
        cond: torch.Tensor,
        **kwargs,
    ):
        """
        cond: (B, T, cond_dim)
        output: (B, T, output_dim)
        """

        print("transformer.py: Transformer.forward()")

        # encoder
        cond_embeddings = self.cond_obs_emb(cond)  # (B,To,n_emb)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[
            :, :tc, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(cond_embeddings + position_embeddings)
        x = self.encoder(x)
        memory = x
        # (B,T_cond,n_emb)

        # decoder
        position_embeddings = self.pos_emb[
            :, : self.horizon, :
        ]  # each position maps to a (learnable) vector
        position_embeddings = position_embeddings.expand(
            cond.shape[0], self.horizon, -1
        )  # repeat for batch dimension
        x = self.drop(position_embeddings)
        # (B,T,n_emb)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask,
        )
        # (B,T,n_emb)

        # head
        x_prehead = self.ln_f(x)
        x = self.head(x_prehead)
        # (B,T,n_out)
        return x, x_prehead


if __name__ == "__main__":

    print("transformer.py: main()")

    transformer = Transformer(
        output_dim=10,
        horizon=4,
        T_cond=1,
        cond_dim=16,
        causal_attn=False,  # no need to use for delta control
        # From Cheng: I found the causal attention masking to be critical to get the transformer variant of diffusion policy to work. My suspicion is that when used without it, the model "cheats" by looking ahead into future end-effector poses, which is almost identical to the action of the current timestep.
        n_cond_layers=0,
    )
    # opt = transformer.configure_optimizers()

    cond = torch_zeros((4, 1, 16))  # B x 1 x cond_dim
    out, _ = transformer(cond)
