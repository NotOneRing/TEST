"""
Soft Actor Critic (SAC) with Diffusion policy.
"""

import logging
from copy import deepcopy


from model.diffusion.diffusion import DiffusionModel

from model.diffusion.sampling import make_timesteps

log = logging.getLogger(__name__)

import tensorflow as tf


from util.torch_to_tf import *


from model.common.mlp import MLP




def is_broadcastable(src, dst):
    print("fax_utils.py:", "is_broadcastable()")
    
    try:
        # use tf.broadcast_static_shape to judge if one shape can be broadcasted to the other
        return tf.broadcast_static_shape(src, dst) == dst
    except ValueError:
        return False







@register_keras_serializable(package="Custom")
class DACER_SinusoidalPosEmb(tf.keras.layers.Layer):
    def __init__(self, dim, name = "SinusoidalPosEmb", **kwargs):

        print("diffusion_DACER_ORIGINAL.py: SinusoidalPosEmb.__init__()")

        super(DACER_SinusoidalPosEmb, self).__init__(name=name,**kwargs)
        self.dim = dim

    def get_config(self):
        """Returns the config of the layer for serialization."""
        config = super(DACER_SinusoidalPosEmb, self).get_config()
        config.update({"dim": self.dim})
        return config

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        return cls(**config)


    def call(self, t, theta: int = 10000, batch_shape = None):
        print("diffusion_DACER_ORIGINAL.py: ", "scaled_sinusoidal_encoding")
        
        assert self.dim % 2 == 0
        if batch_shape is not None:
            assert is_broadcastable(t.shape, batch_shape)

        scale = 1 / self.dim ** 0.5
        half_dim = self.dim // 2
        freq_seq = torch_arange(start = 0, end = half_dim, step=1) / half_dim
        inv_freq = theta ** -freq_seq

        emb = tf.einsum('..., j -> ... j', t, inv_freq)

        emb = torch_cat([tf.sin(emb), tf.cos(emb)], dim=-1)

        emb *= scale

        if batch_shape is not None:
            emb = tf.broadcast_to(emb, (*batch_shape, self.dim))

        return emb



class DACERPolicyNet(tf.keras.Model):
    def __init__(self, action_dim, env_name, activation = "GELU", hidden_sizes = (256, 256, 256), 
                 output_activation = "Identity", time_dim = 16, 
                 name = None):
        self.time_dim = time_dim
        
        if self.env_name == "hopper-medium-v2":
            # shape1 = (128, 4, 3)
            shape2 = (128, 1, 11)
        elif self.env_name == "kitchen-complete-v0":
            # shape1 = (128, 4, 9)
            shape2 = (128, 1, 60)
        elif self.env_name == "kitchen-mixed-v0":
            # shape1 = (256, 4, 9)
            shape2 = (256, 1, 60)
        elif self.env_name == "kitchen-partial-v0":
            # shape1 = (128, 4, 9)
            shape2 = (128, 1, 60)
        elif self.env_name == "walker2d-medium-v2":
            # shape1 = (128, 4, 6)
            shape2 = (128, 1, 17)
        elif self.env_name == "halfcheetah-medium-v2":
            # shape1 = (128, 4, 6)
            shape2 = (128, 1, 17)
        elif self.env_name == "lift":
            # shape1 = (256, 4, 7)
            shape2 = (256, 1, 19)
        elif self.env_name == "can":
            # shape1 = (256, 4, 7)
            shape2 = (256, 1, 23)
        elif self.env_name == "square":
            # shape1 = (256, 4, 7)
            shape2 = (256, 1, 23)
        elif self.env_name == "transport":
            # shape1 = (256, 8, 14)
            shape2 = (256, 1, 59)

        self.time_embedding = nn_Sequential([
                DACER_SinusoidalPosEmb(dim=self.time_dim, batch_shape=obs.shape[:-1]),
                nn_Linear(time_dim, time_dim * 2, name_Dense = "DiffusionMLP_time_embedding_1"),
                nn_Mish(),
                nn_Linear(time_dim * 2, time_dim, name_Dense = "DiffusionMLP_time_embedding_2"),        
        ], name = "nn_Sequential_time_embedding")

        self.hidden_sizes = hidden_sizes
        self.action_dim = action_dim

        dim_list = list(*self.hidden_sizes) + [self.action_dim]
        self.mlp =  MLP(dim_list = dim_list, activation_type=self.activation, out_activation_type=self.output_activation)


    def call(self, obs, act, t):
        # act_dim = act.shape[-1]
        # te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        # te = hk.Linear(self.time_dim * 2)(te)
        # te = self.activation(te)
        # te = hk.Linear(self.time_dim)(te)
        te = self.time_embedding(t)
        inputs = torch_cat((obs, act, te), dim=-1)
        return self.mlp(inputs)









class DACER_Original_Diffusion(DiffusionModel):
    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):

        print("diffusion_DACER_ORIGINAL.py: DACER_Original_Diffusion.__init__()")

        super().__init__(network=actor, **kwargs)

        # initialize doubel critic networks
        self.critic = critic

        self.build_actor(self.critic)

        # initialize double target networks
        self.target_critic = deepcopy(self.critic)


        self.actor = DACERPolicyNet()

        self.build_actor(self.actor)



        self.step = 0

        self.delay_alpha_update = 10000

        self.delay_update = 2
        
        self.tau = 0.005

        self.mean_q1_std = -1.0
        self.mean_q2_std = -1.0
        self.entropy = 0.0


    

    def get_action(self, obs, alpha):
    
        action = self.call(
            cond=obs,
            deterministic=False
        )
        action = action + tf.random.normal(action.shape) * alpha * 0.15 # other envs 0.1
        action = tf.clip_by_value(action, -1, 1)
        return action



    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        rewards,
        terminated,
        gamma,
        alpha,
    ):
    
        print("diffusion_DACER_ORIGINAL.py: diffusion_DACER_ORIGINAL.loss_critic()")

        self.reward_scale = 0.2


        next_actions = self.get_action(next_obs, alpha)

        next_q1_mean, next_q1_std, next_q2_mean, next_q2_std = self.target_critic(
            next_obs,
            next_actions,
        )

        z1 = tf.random.normal(next_q1_mean.shape)
        z1 = tf.clip_by_value(z1, -3.0, 3.0)

        z2 = tf.random.normal(next_q2_mean.shape)
        z2 = tf.clip_by_value(z2, -3.0, 3.0)

        next_q1_sample = next_q1_mean + next_q1_std * z1
        next_q2_sample = next_q2_mean + next_q2_std * z2

        cur_rewards = rewards * self.reward_scale

        next_q_mean = torch_min(next_q1_mean, other=next_q2_mean)
        next_q_sample = tf.where(next_q1_mean < next_q2_mean, next_q1_sample, next_q2_sample)

        q_target = next_q_mean
        q_target_sample = next_q_sample

        q_backup = cur_rewards + (1 - terminated) * gamma * q_target
        q_backup_sample = cur_rewards + (1 - terminated) * gamma * q_target_sample


        B = tf.shape(obs["state"])[0]
        reshape_state = torch_tensor_view(obs["state"], [B, -1])
        reshape_actions = torch_tensor_view(actions, [B, -1])
        x = torch_cat((reshape_state, reshape_actions), dim=-1)
        
        def q_loss_fn(q_network, mean_q_std: float):
            cur_x = deepcopy(x)
            q_result = q_network(cur_x)
            q_mean, q_std = q_result[..., 0], q_result[..., 1]

            new_mean_q_std = tf.reduce_mean(q_std)
            mean_q_std = tf.stop_gradient(
                int(mean_q_std == -1.0) * new_mean_q_std +
                int(mean_q_std != -1.0) * (self.tau * new_mean_q_std + (1 - self.tau) * mean_q_std)
            )
            q_backup_bounded = tf.stop_gradient(q_mean + tf.clip_by_value(q_backup_sample - q_mean, -3 * mean_q_std, 3 * mean_q_std))

            q_std_detach = tf.stop_gradient( torch_max(q_std, other = 0.0))
            epsilon = 0.1
            q_loss = -(mean_q_std ** 2 + epsilon) * tf.reduce_mean(
                q_mean * tf.stop_gradient(q_backup - q_mean) / (q_std_detach ** 2 + epsilon) +
                q_std * (( tf.stop_gradient(q_mean) - q_backup_bounded) ** 2 
                - q_std_detach ** 2) / (q_std_detach ** 3 + epsilon)
            )
            return q_loss, (q_mean, q_std, mean_q_std)

        (q1_loss, (q1_mean, q1_std, mean_q1_std)) = q_loss_fn( self.critic.Q1, self.mean_q1_std)
        (q2_loss, (q2_mean, q2_std, mean_q2_std)) = q_loss_fn( self.critic.Q2, self.mean_q2_std)

        self.mean_q1_std = mean_q1_std
        self.mean_q2_std = mean_q2_std


        return q1_loss, q2_loss






    def loss_actor(self, obs, alpha):

        print("diffusion_DACER_ORIGINAL.py: diffusion_DACER_ORIGINAL.loss_actor()")


        action = self.get_action(obs, alpha)


        current_q1, _, current_q2, _ = self.critic(obs, action)

        loss_actor = -torch_min(current_q1, current_q2)


        return torch_mean(loss_actor)
    





    # @tf.function
    def call(self, cond, deterministic=False):
        """Modifying denoising schedule"""


        print("diffusion_DACER_ORIGINAL.py: diffusion_DACER_ORIGINAL.forward()")

        with torch_no_grad() as tape:

            B = cond["state"].shape[0]

            x = tf.random.normal( (B, self.horizon_steps, self.action_dim) )

            t_all = list(reversed(range(self.denoising_steps)))
            for i, t in enumerate(t_all):
                t_b = make_timesteps(B, t)
                

                mean, logvar = self.p_mean_var(
                    x=x,
                    t=t_b,
                    cond_state=cond['state'],
                )

                std = torch_exp(0.5 * logvar)


                # Add noise
                noise = torch_randn_like(x)
                x = mean + std * noise

                # Clamp action at final step
                if self.final_action_clip_value is not None and i == len(t_all) - 1:
                    x = torch_clamp(x, -self.final_action_clip_value, self.final_action_clip_value)
            
        return x



    def estimate_entropy(self, actions, num_components=3):  # (batch, sample, dim)
        import numpy as np
        from sklearn.mixture import GaussianMixture
        total_entropy = []
        

        shape = actions.shape
        actions = actions.reshape(shape[0], shape[1], -1)

        for action in actions:
            gmm = GaussianMixture(n_components=num_components, covariance_type='full')
            gmm.fit(action)
            weights = gmm.weights_
            entropies = []
            for i in range(gmm.n_components):
                cov_matrix = gmm.covariances_[i]
                d = cov_matrix.shape[0]
                entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(cov_matrix)[1]
                entropies.append(entropy)
            entropy = -np.sum(weights * np.log(weights)) + np.sum(weights * np.array(entropies))
            total_entropy.append(entropy)

        total_entropy_sum = sum(total_entropy)
        total_entropy_len = len(total_entropy)

        return total_entropy_sum, total_entropy_len


    def cal_entropy(self, obs, alpha, num_samples):

        actions = []
        for i in range(num_samples):
            cur_actions = self.get_action(obs, alpha)
            actions.append(cur_actions)
        
        actions = np.stack(actions, axis=0)

        shape = actions.shape
        new_order = [1, 0] + list(range(2, len(shape)))
        actions = np.transpose(actions, new_order)


        entropy_sum, entropy_len = self.estimate_entropy( actions )
        
        return entropy_sum, entropy_len



    def loss_temperature(self, obs, alpha, target_entropy):

        print("diffusion_DACER_ORIGINAL.py: diffusion_DACER_ORIGINAL.loss_temperature()")

        self.num_samples = 200

        prev_entropy = self.entropy if hasattr(self, 'entropy') else tf.float32(0.0)

        if self.step % self.delay_alpha_update == 0:
            entropy_sum_list = []
            entropy_len_list = []
            B = obs['state'].shape[0]


            # parallel for batch dim
            cur_obs_dict = obs
            entropy_sum, entropy_len = self.cal_entropy(cur_obs_dict, alpha, self.num_samples)
            entropy_sum_list.append(entropy_sum)
            entropy_len_list.append(entropy_len)


            self.entropy = sum(entropy_sum_list) / sum(entropy_len_list)
        else:
            self.entropy = prev_entropy

        print("self.entropy = ", self.entropy)

        loss_alpha = -torch_mean( torch_log(alpha) * ( -self.entropy + target_entropy ) )


        return loss_alpha



    def update_target_critic(self, tau):

        print("diffusion_DACER_ORIGINAL.py: diffusion_DACER_ORIGINAL.update_target_critic()")

        critic_variables = self.critic.trainable_variables
        target_critic_variables = self.target_critic.trainable_variables

        for target_param, source_param in zip(target_critic_variables, critic_variables):
            target_param.assign(target_param * (1.0 - tau) + source_param * tau)










































# def jax_lax_stop_gradient(tensor):
#     return tf.stop_gradient(tensor)


# def jnp_exp(input):
#     return torch_exp(input)


# def jnp_mean(input_tensor1, dim = None, keepdim = False):
#     return torch_mean(input_tensor1, dim, keepdim)
    

# def jnp_maximum(input, dim = None, other = None):
#     return torch_max(input, dim, other)


# def jnp_minimum(input, dim = None, other = None):
#     return torch_min(input, dim, other)


# def jnp_clip(input, min = float('-inf'), max = float('inf'), out=None):
#     return torch_clip(input, min, max, out)


# def jnp_where(index_tensor, input_tensor = None, replace_value = None):
#     return torch_where(index_tensor, input_tensor, replace_value)



# def jnp_float32(input):
#     output = tf.cast(input, tf.float32)
#     return output



# def jax_value_and_grad():
#     pass



# def jax_pure_callback():
#     pass



# def jax_lax_cond():
#     pass



# def optax_apply_updates():
#     pass









#         def stateless_update(
#             key: jax.Array, state: DACERTrainState, data: Experience
#         ) -> Tuple[DACERTrainState, Metric]:
#             obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
#             q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha = state.params
#             q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
#             step, mean_q1_std, mean_q2_std = state.step, state.mean_q1_std, state.mean_q2_std
#             next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key, log_alpha_key = jax.random.split(key, 5)

#             reward *= self.reward_scale

#             # compute target q
#             next_action = self.agent.get_action(next_eval_key, (policy_params, log_alpha), next_obs)
#             next_q1_mean, _, next_q1_sample = self.agent.q_evaluate(new_q1_eval_key, target_q1_params, next_obs, next_action)
#             next_q2_mean, _, next_q2_sample = self.agent.q_evaluate(new_q2_eval_key, target_q2_params, next_obs, next_action)
#             next_q_mean = jnp_minimum(next_q1_mean, next_q2_mean)
#             next_q_sample = jnp_where(next_q1_mean < next_q2_mean, next_q1_sample, next_q2_sample)
#             q_target = next_q_mean 
#             q_target_sample = next_q_sample 
#             q_backup = reward + (1 - done) * self.gamma * q_target
#             q_backup_sample = reward + (1 - done) * self.gamma * q_target_sample

#             # update q
#             def q_loss_fn(q_params: hk.Params, mean_q_std: float) -> jax.Array:
#                 q_mean, q_std = self.agent.q(q_params, obs, action)
#                 new_mean_q_std = jnp_mean(q_std)
#                 mean_q_std = jax_lax_stop_gradient(
#                     (mean_q_std == -1.0) * new_mean_q_std +
#                     (mean_q_std != -1.0) * (self.tau * new_mean_q_std + (1 - self.tau) * mean_q_std)
#                 )
#                 q_backup_bounded = jax_lax_stop_gradient(q_mean + jnp_clip(q_backup_sample - q_mean, -3 * mean_q_std, 3 * mean_q_std))
#                 q_std_detach = jax_lax_stop_gradient(jnp_maximum(q_std, 0))
#                 epsilon = 0.1
#                 q_loss = -(mean_q_std ** 2 + epsilon) * jnp.mean(
#                     q_mean * jax_lax_stop_gradient(q_backup - q_mean) / (q_std_detach ** 2 + epsilon) +
#                     q_std * ((jax_lax_stop_gradient(q_mean) - q_backup_bounded) ** 2 - q_std_detach ** 2) / (q_std_detach ** 3 + epsilon)
#                 )
#                 return q_loss, (q_mean, q_std, mean_q_std)

#             (q1_loss, (q1_mean, q1_std, mean_q1_std)), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params, mean_q1_std)
#             (q2_loss, (q2_mean, q2_std, mean_q2_std)), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params, mean_q2_std)
            
#             def cal_entropy():
#                 keys = jax.random.split(log_alpha_key, self.num_samples)
#                 actions = jax.vmap(self.agent.get_action, in_axes=(0, None, None), out_axes=1)(keys, (policy_params, jax.lax.stop_gradient(log_alpha)), obs)
#                 entropy = jax.pure_callback(estimate_entropy, jax.ShapeDtypeStruct((), jnp.float32), actions)
#                 entropy = jax_lax_stop_gradient(entropy)
#                 return entropy
            
#             prev_entropy = state.entropy if hasattr(state, 'entropy') else jnp_float32(0.0)
            
#             entropy = jax_lax_cond(
#                 step % self.delay_alpha_update == 0,
#                 cal_entropy,
#                 lambda: prev_entropy
#             )
            
#             # update policy
#             def policy_loss_fn(policy_params) -> jax.Array:
#                 new_action = self.agent.get_action(new_eval_key, (policy_params, log_alpha), obs)
#                 q1_mean, _ = self.agent.q(q1_params, obs, new_action)
#                 q2_mean, _ = self.agent.q(q2_params, obs, new_action)
#                 q_mean = jnp_minimum(q1_mean, q2_mean)
#                 policy_loss = jnp_mean(-q_mean) 
#                 return policy_loss

#             total_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)
            
#             # update alpha
#             def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
#                 log_alpha_loss = -jnp.mean(log_alpha * (-entropy + self.agent.target_entropy))
#                 return log_alpha_loss

#             # update networks
#             def param_update(optim, params, grads, opt_state):
#                 update, new_opt_state = optim.update(grads, opt_state)
#                 new_params = optax.apply_updates(params, update)
#                 return new_params, new_opt_state

#             def delay_param_update(optim, params, grads, opt_state):
#                 return jax.lax.cond(
#                     step % self.delay_update == 0,
#                     lambda params, opt_state: param_update(optim, params, grads, opt_state),
#                     lambda params, opt_state: (params, opt_state),
#                     params, opt_state
#                 )
                
#             def delay_alpha_param_update(optim, params, opt_state):
#                 return jax.lax.cond(
#                     step % self.delay_alpha_update == 0,
#                     lambda params, opt_state: param_update(optim, params, jax.grad(log_alpha_loss_fn)(params), opt_state),
#                     lambda params, opt_state: (params, opt_state),
#                     params, opt_state
#                 )
                
#             def delay_target_update(params, target_params, tau):
#                 return jax.lax.cond(
#                     step % self.delay_update == 0,
#                     lambda target_params: optax.incremental_update(params, target_params, tau),
#                     lambda target_params: target_params,
#                     target_params
#                 )

#             q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
#             q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
#             policy_params, policy_opt_state = delay_param_update(self.optim, policy_params, policy_grads, policy_opt_state)
#             log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

#             target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
#             target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)

#             state = DACERTrainState(
#                 params=DACERParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha),
#                 opt_state=DACEROptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state),
#                 step=step + 1,
#                 mean_q1_std=mean_q1_std,
#                 mean_q2_std=mean_q2_std,
#                 entropy=entropy,
#             )

            
#             info = {
#                 "q1_loss": q1_loss,
#                 "q1_mean": jnp.mean(q1_mean),
#                 "q1_std": jnp.mean(q1_std),
#                 "q2_loss": q2_loss,
#                 "q2_mean": jnp.mean(q2_mean),
#                 "q2_std": jnp.mean(q2_std),
#                 "policy_loss": total_loss,
#                 "alpha": jnp.exp(log_alpha),
#                 "mean_q1_std": mean_q1_std,
#                 "mean_q2_std": mean_q2_std,
#                 "entropy": entropy,
#             }
#             return state, info

#         self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

#     def get_policy_params(self):
#         return (self.state.params.policy, self.state.params.log_alpha)




























