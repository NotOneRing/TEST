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

        self.build_critic(self.critic)

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

        return q1_loss, q2_loss






    def loss_actor(self, obs, alpha):

        print("diffusion_sac.py: SAC_Diffusion.loss_actor()")

        action = self.get_action(obs, alpha)

        print("loss_actor: action = ", action)


        current_q1, _, current_q2, _ = self.critic(obs, action)

        if self.output_actor_once_flag:
            print("loss_actor: tf.reduce_mean(current_q1) = ", tf.reduce_mean(current_q1) )
            print("loss_actor: tf.reduce_mean(current_q2) = ", tf.reduce_mean(current_q2) )

        loss_actor = -torch_min(current_q1, current_q2)

        if self.output_actor_once_flag:
            print("loss_actor: tf.reduce_mean(loss_actor) = ", tf.reduce_mean(loss_actor) )


        return torch_mean(loss_actor)
    





    def call(self, cond, deterministic=False):
        """Modifying denoising schedule"""


        print("diffusion_DACER.py: DACERDiffusion.forward()")

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

        print("cal_entropy(): actions = ", actions)

        shape = actions.shape
        new_order = [1, 0] + list(range(2, len(shape)))
        actions = np.transpose(actions, new_order)


        entropy_sum, entropy_len = self.estimate_entropy( actions )
        
        return entropy_sum, entropy_len



    def loss_temperature(self, obs, alpha, target_entropy):

        print("diffusion_sac.py: SAC_Diffusion.loss_temperature()")

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

        if self.output_alpha_once_flag:
            print("loss_alpha = ", loss_alpha)

        return loss_alpha



    def update_target_critic(self, tau):

        print("diffusion_sac.py: SAC_Diffusion.update_target_critic()")

        critic_variables = self.critic.trainable_variables
        target_critic_variables = self.target_critic.trainable_variables

        for target_param, source_param in zip(target_critic_variables, critic_variables):
            target_param.assign(target_param * (1.0 - tau) + source_param * tau)







    def build_critic(self, critic, shape1=None, shape2=None):
    
        print("build_critic: self.env_name = ", self.env_name)

        if shape1 != None and shape2 != None:
            pass
        # Gym - hopper/walker2d/halfcheetah
        elif self.env_name == "hopper-medium-v2":
            shape1 = (128, 4, 3)
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
            shape2 = (128, 1, 17)
        # Robomimic - lift/can/square/transport
        elif self.env_name == "lift":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 19)

        elif self.env_name == "can":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 23)

        elif self.env_name == "square":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 23)

        elif self.env_name == "transport":
            shape1 = (256, 8, 14)
            shape2 = (256, 1, 59)

        # the same name "avoiding-m5" for D3IL with avoid_m1/m2/m3
        elif self.env_name == "avoiding-m5" or self.env_name == "avoid":
            shape1 = (16, 4, 2)
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
            raise RuntimeError("The build shape is not implemented for current dataset")


        if OUTPUT_VARIABLES:
            print("type(shape1) = ", type(shape1))
            print("type(shape2) = ", type(shape2))

            print("shape1 = ", shape1)
            print("shape2 = ", shape2)


        param1 = torch_ones(*shape1)
        param2 = torch_ones(*shape2)

        build_dict = {'state': param2}


        
        next_q1_mean, next_q1_std, next_q2_mean, next_q2_std = critic(
            build_dict,
            param1,
        )
        
        
        print("all_one_build_result next_q1_mean = ", sum(next_q1_mean))
        print("all_one_build_result next_q1_std = ", sum(next_q1_mean))
        print("all_one_build_result next_q2_mean = ", sum(next_q1_mean))
        print("all_one_build_result next_q2_std = ", sum(next_q1_mean))



































