"""
Soft Actor Critic (SAC) with Diffusion policy.
"""

import logging
from copy import deepcopy


from model.diffusion.diffusion import DiffusionModel

from model.diffusion.sampling import make_timesteps

log = logging.getLogger(__name__)

# from util.torch_to_tf import torch_mse_loss, torch_min, torch_mean

# from util.torch_to_tf import torch_no_grad, torch_mean

import tensorflow as tf


from util.torch_to_tf import *





class DACER_Diffusion(DiffusionModel):
    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):

        print("diffusion_sac.py: SAC_Diffusion.__init__()")

        super().__init__(network=actor, **kwargs)

        # initialize doubel critic networks
        self.critic = critic
        # .to(self.device)

        self.build_critic(self.critic)

        # initialize double target networks
        # self.target_critic = deepcopy(self.critic)

        self.target_critic = tf.keras.models.clone_model(self.critic)
        self.build_critic(self.target_critic)
        self.target_critic.set_weights(self.critic.get_weights())


        self.actor = self.network



        self.step = 0

        self.delay_alpha_update = 10000

        self.delay_update = 2
        
        self.tau = 0.005

        self.mean_q1_std = -1.0
        self.mean_q2_std = -1.0
        self.entropy = 0.0


        self.output_critic_once_flag = True
        self.output_actor_once_flag = True
        self.output_alpha_once_flag = True

    

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
    
        print("diffusion_sac.py: SAC_Diffusion.loss_critic()")

        self.reward_scale = 0.2

        # with torch_no_grad() as tape:

        next_actions = self.get_action(next_obs, alpha)

        print("loss_critic: next_actions = ", next_actions)

        next_q1_mean, next_q1_std, next_q2_mean, next_q2_std = self.critic(
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
            
            if self.output_critic_once_flag:
                print("q_mean.shape = ", q_mean.shape)
                print("q_std.shape = ", q_std.shape)

                print("loss_critic: tf.reduce_mean(q_mean) = ", tf.reduce_mean(q_mean))

            new_mean_q_std = tf.reduce_mean(q_std)

            if self.output_critic_once_flag:
                print("loss_critic: new_mean_q_std = ", new_mean_q_std)

            mean_q_std = tf.stop_gradient(
                int(mean_q_std == -1.0) * new_mean_q_std +
                int(mean_q_std != -1.0) * (self.tau * new_mean_q_std + (1 - self.tau) * mean_q_std)
            )

            if self.output_critic_once_flag:
                print("loss_critic: mean_q_std = ", mean_q_std)

            q_backup_bounded = tf.stop_gradient(q_mean + tf.clip_by_value(q_backup_sample - q_mean, -3 * mean_q_std, 3 * mean_q_std))

            if self.output_critic_once_flag:
                print("loss_critic: tf.reduce_mean(q_backup_bounded) = ", tf.reduce_mean(q_backup_bounded))

            q_std_detach = tf.stop_gradient( torch_max(q_std, other = 0.0))

            if self.output_critic_once_flag:
                print("loss_critic: tf.reduce_mean(q_std_detach) = ", tf.reduce_mean(q_std_detach))

            epsilon = 0.1
            q_loss = -(mean_q_std ** 2 + epsilon) * tf.reduce_mean(
                q_mean * tf.stop_gradient(q_backup - q_mean) / (q_std_detach ** 2 + epsilon) +
                q_std * ( ( tf.stop_gradient(q_mean) - q_backup_bounded) ** 2 
                - q_std_detach ** 2 ) / (q_std_detach ** 3 + epsilon)
            )

            if self.output_critic_once_flag:
                print("loss_critic: q_loss = ", q_loss)

            return q_loss, (q_mean, q_std, mean_q_std)

        (q1_loss, (q1_mean, q1_std, mean_q1_std)) = q_loss_fn( self.critic.Q1, self.mean_q1_std)
        (q2_loss, (q2_mean, q2_std, mean_q2_std)) = q_loss_fn( self.critic.Q2, self.mean_q2_std)

        if self.output_critic_once_flag:
            print("loss_critic: q1_loss = ", q1_loss)
            print("loss_critic: q2_loss = ", q2_loss)
            print("loss_critic: mean_q1_std = ", mean_q1_std)
            print("loss_critic: mean_q2_std = ", mean_q2_std)


        self.mean_q1_std = mean_q1_std
        self.mean_q2_std = mean_q2_std

        # self.output_critic_once_flag = False

        return q1_loss, q2_loss






    def loss_actor(self, obs, alpha):

        print("diffusion_sac.py: SAC_Diffusion.loss_actor()")

        # action = self.call(
        #     obs,
        #     deterministic=False
        # )
        action = self.get_action(obs, alpha)

        print("loss_actor: action = ", action)


        # print("loss_actor: action = ", action)

        current_q1, _, current_q2, _ = self.critic(obs, action)

        if self.output_actor_once_flag:
            print("loss_actor: tf.reduce_mean(current_q1) = ", tf.reduce_mean(current_q1) )
            print("loss_actor: tf.reduce_mean(current_q2) = ", tf.reduce_mean(current_q2) )

        loss_actor = -torch_min(current_q1, current_q2)

        if self.output_actor_once_flag:
            print("loss_actor: tf.reduce_mean(loss_actor) = ", tf.reduce_mean(loss_actor) )

        # print("loss_actor: loss_actor = ", loss_actor)

        # self.output_actor_once_flag = False

        return torch_mean(loss_actor)
    





    # @tf.function
    def call(self, cond, deterministic=False):
        """Modifying denoising schedule"""


        print("diffusion_DACER.py: DACERDiffusion.forward()")

        with torch_no_grad() as tape:

            B = cond["state"].shape[0]

            x = tf.random.normal( (B, self.horizon_steps, self.action_dim) )

            # print("self.horizon_steps = ", self.horizon_steps)
            # print("self.action_dim = ", self.action_dim)

            t_all = list(reversed(range(self.denoising_steps)))
            for i, t in enumerate(t_all):
                t_b = make_timesteps(B, t)
                
                # print("t_b.shape = ", t_b.shape)

                mean, logvar = self.p_mean_var(
                    x=x,
                    t=t_b,
                    # cond=cond,
                    cond_state=cond['state'],
                )

                std = torch_exp(0.5 * logvar)

                # # Determine noise level
                # if deterministic and t == 0:
                #     std = torch_zeros_like(std)
                # elif deterministic:
                #     std = torch_clip(std, 1e-3, float('inf'))
                # else:
                #     std = torch_clip(std, self.min_sampling_denoising_std, float('inf'))

                # Add noise
                noise = torch_randn_like(x)
                # noise = torch_clamp(noise, -self.randn_clip_value, self.randn_clip_value)
                x = mean + std * noise

                # Clamp action at final step
                if self.final_action_clip_value is not None and i == len(t_all) - 1:
                    x = torch_clamp(x, -self.final_action_clip_value, self.final_action_clip_value)
            
        return x



    def estimate_entropy(self, actions, num_components=3):  # (batch, sample, dim)
        import numpy as np
        from sklearn.mixture import GaussianMixture
        total_entropy = []
        # total_entropy_sum = []
        # total_entropy_len = []
        

        shape = actions.shape
        actions = actions.reshape(shape[0], shape[1], -1)

        # print("estimate_entropy: actions.shape = ", actions.shape)

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

        # final_entropy = sum(total_entropy) / len(total_entropy)
        # return final_entropy
        total_entropy_sum = sum(total_entropy)
        total_entropy_len = len(total_entropy)

        return total_entropy_sum, total_entropy_len


    def cal_entropy(self, obs, alpha, num_samples):

        actions = []
        for i in range(num_samples):
            cur_actions = self.get_action(obs, alpha)
            actions.append(cur_actions)
        
        actions = np.stack(actions, axis=0)

        print("cal_entropy(): actions = ", actions)

        shape = actions.shape
        new_order = [1, 0] + list(range(2, len(shape)))
        actions = np.transpose(actions, new_order)


        # print("actions = ", actions)
        # print("actions.shape = ", actions.shape)
        # print("type(actions) = ", type(actions))


        entropy_sum, entropy_len = self.estimate_entropy( actions )
        
        return entropy_sum, entropy_len



    def loss_temperature(self, obs, alpha, target_entropy):

        print("diffusion_sac.py: SAC_Diffusion.loss_temperature()")

        self.num_samples = 200
        # self.num_samples = 20

        prev_entropy = self.entropy if hasattr(self, 'entropy') else tf.float32(0.0)

        if self.step % self.delay_alpha_update == 0:
            entropy_sum_list = []
            entropy_len_list = []
            B = obs['state'].shape[0]

            # print("obs['state'].shape = ", obs['state'].shape)
            # print( "type(obs['state']) = ", type(obs['state']) )

            #one by one
            # for i in range(B):
            #     cur_obs = tf.gather(obs['state'], i, axis=0)
            #     cur_obs = torch_unsqueeze(cur_obs, 0)
            #     # print("cur_obs.shape = ", cur_obs.shape)
            #     cur_obs = torch_squeeze(cur_obs, 1)
            #     # print("cur_obs.shape = ", cur_obs.shape)
            #     cur_obs_dict = {'state': cur_obs}

            #     entropy_sum, entropy_len = self.cal_entropy(cur_obs_dict, alpha, self.num_samples)
            #     entropy_sum_list.append(entropy_sum)
            #     entropy_len_list.append(entropy_len)

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

        if self.output_alpha_once_flag:
            print("loss_alpha = ", loss_alpha)

        # self.output_alpha_once_flag = False

        return loss_alpha



    def update_target_critic(self, tau):

        print("diffusion_sac.py: SAC_Diffusion.update_target_critic()")

        critic_variables = self.critic.trainable_variables
        target_critic_variables = self.target_critic.trainable_variables

        for target_param, source_param in zip(target_critic_variables, critic_variables):
            target_param.assign(target_param * (1.0 - tau) + source_param * tau)







    def build_critic(self, critic, shape1=None, shape2=None):
        # return
    
        print("build_critic: self.env_name = ", self.env_name)

        if shape1 != None and shape2 != None:
            pass
        # Gym - hopper/walker2d/halfcheetah
        elif self.env_name == "hopper-medium-v2":
            # hopper_medium
            # item_actions_copy.shape =  
            shape1 = (128, 4, 3)
            # cond_copy['state'].shape =  
            shape2 = (128, 1, 11)
        elif self.env_name == "kitchen-complete-v0":
            shape1 = (128, 4, 9)
            shape2 = (128, 1, 60)
        elif self.env_name == "kitchen-mixed-v0":
            shape1 = (256, 4, 9)
            shape2 = (256, 1, 60)
        elif self.env_name == "kitchen-partial-v0":
            shape1 = (128, 4, 9)
            shape2 = (128, 1, 60)
        elif self.env_name == "walker2d-medium-v2":
            shape1 = (128, 4, 6)
            shape2 = (128, 1, 17)
        elif self.env_name == "halfcheetah-medium-v2":
            shape1 = (128, 4, 6)
            # shape1 = (128, 1, 6)
            shape2 = (128, 1, 17)
        # Robomimic - lift/can/square/transport
        elif self.env_name == "lift":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 19)

        elif self.env_name == "can":
            #can 
            # item_actions_copy.shape =  
            shape1 = (256, 4, 7)
            # cond_copy['state'].shape =  
            shape2 = (256, 1, 23)

        elif self.env_name == "square":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 23)

        elif self.env_name == "transport":
            shape1 = (256, 8, 14)
            shape2 = (256, 1, 59)

        # the same name "avoiding-m5" for D3IL with avoid_m1/m2/m3
        elif self.env_name == "avoiding-m5" or self.env_name == "avoid":
            #avoid_m1
            # item_actions_copy.shape =  
            shape1 = (16, 4, 2)
            # cond_copy['state'].shape =  
            shape2 = (16, 1, 4)

        # Furniture-Bench - one_leg/lamp/round_table_low/med
        elif self.env_name == "lamp_low_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 44)
        elif self.env_name == "lamp_med_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 44)
        elif self.env_name == "one_leg_low_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 58)
        elif self.env_name == "one_leg_med_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 58)
        elif self.env_name == "round_table_low_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 44)
        elif self.env_name == "round_table_med_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 44)
        
        else:
            # #one_leg_low
            # # item_actions_copy.shape =  
            # shape1 = (256, 8, 10)
            # # cond_copy['state'].shape =  
            # shape2 = (256, 1, 58)
            raise RuntimeError("The build shape is not implemented for current dataset")


        # param1 = tf.constant(np.random.randn(*shape1).astype(np.float32))
        # param2 = tf.constant(np.random.randn(*shape2).astype(np.float32))


        if OUTPUT_VARIABLES:
            print("type(shape1) = ", type(shape1))
            print("type(shape2) = ", type(shape2))

            print("shape1 = ", shape1)
            print("shape2 = ", shape2)


        param1 = torch_ones(*shape1)
        param2 = torch_ones(*shape2)

        build_dict = {'state': param2}


        
        # _ = self.loss_ori(param1, build_dict)
        # all_one_build_result = 
        next_q1_mean, next_q1_std, next_q2_mean, next_q2_std = critic(
            build_dict,
            param1,
        )
        # self.loss_ori_build(actor, training=False, x_start = param1, cond=build_dict)

        print("all_one_build_result next_q1_mean = ", sum(next_q1_mean))
        print("all_one_build_result next_q1_std = ", sum(next_q1_mean))
        print("all_one_build_result next_q2_mean = ", sum(next_q1_mean))
        print("all_one_build_result next_q2_std = ", sum(next_q1_mean))






































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




























