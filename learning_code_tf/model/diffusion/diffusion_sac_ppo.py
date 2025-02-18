"""
DPPO: Diffusion Policy Policy Optimization. 

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""

from typing import Optional

import logging
import math

import tensorflow as tf

# from util.torch_to_tf import torch_quantile
# # import tensorflow_probability as tfp

# from util.torch_to_tf import torch_no_grad, torch_tensor_item

# from util.torch_to_tf import torch_clamp, torch_exp, torch_mean, torch_max, torch_tensor_float, torch_abs, torch_tensor, torch_std, torch_tensor_view

from util.torch_to_tf import *

from model.diffusion.diffusion import DiffusionModel, Sample
from model.diffusion.sampling import make_timesteps, extract



import numpy as np


log = logging.getLogger(__name__)
from model.diffusion.diffusion_vpg import VPGDiffusion


from util.config import DEBUG, NP_RANDOM, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, OUTPUT_POSITIONS, OUTPUT_FUNCTION_HEADER


class SAC_PPODiffusion(VPGDiffusion):
    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0,
        clip_advantage_upper_quantile: float = 1,
        norm_adv: bool = True,
        **kwargs,
    ):

        print("diffusion_ppo.py: PPODiffusion.__init__()")

        super().__init__(**kwargs)

        # Whether to normalize advantages within batch
        self.norm_adv = norm_adv

        # Clipping value for policy loss
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate

        # Clipping value for value loss
        self.clip_vloss_coef = clip_vloss_coef

        # Discount factor for diffusion MDP
        self.gamma_denoising = gamma_denoising

        # Quantiles for clipping advantages
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile


        # self.build_critic(self.critic)

        # self.target_critic = tf.keras.models.clone_model(self.critic)
        # self.build_critic(self.target_critic)
        # self.target_critic.set_weights(self.critic.get_weights())





        self.step_count = 0

        self.delay_alpha_update = 10000

        self.delay_update = 2
        
        self.tau = 0.005

        self.mean_q1_std = -1.0
        self.mean_q2_std = -1.0
        self.entropy = 0.0

        self.alpha = tf.Variable(np.array([0.272]), dtype=tf.float32)


        if self.env_name == "hopper-medium-v2":
            self.target_entropy = -4 * 3
        elif self.env_name == "kitchen-complete-v0":
            self.target_entropy = -4 * 9
        elif self.env_name == "kitchen-mixed-v0":
            self.target_entropy = -4 * 9
        elif self.env_name == "kitchen-partial-v0":
            self.target_entropy = -4 * 9
        elif self.env_name == "walker2d-medium-v2":
            self.target_entropy = -4 * 6
        elif self.env_name == "halfcheetah-medium-v2":
            self.target_entropy = -4 * 6
        elif self.env_name == "lift":
            self.target_entropy = -4 * 7
        elif self.env_name == "can":
            self.target_entropy = -4 * 7
        elif self.env_name == "square":
            self.target_entropy = -4 * 7
        elif self.env_name == "transport":
            self.target_entropy = -8 * 14
        elif self.env_name == "avoiding-m5" or self.env_name == "avoid":
            self.target_entropy = -4 * 2

        







    def get_config(self):
        """
        Returns the configuration of the PPODiffusion instance as a dictionary.
        
        Returns:
            dict: Configuration dictionary for the PPODiffusion instance.
        """
        # Get the config from the parent class (VPGDiffusion)
        config = super().get_config()
        
        # Add the configuration for PPODiffusion-specific attributes
        config.update({
            'gamma_denoising': self.gamma_denoising,
            'clip_ploss_coef': self.clip_ploss_coef,
            'clip_ploss_coef_base': self.clip_ploss_coef_base,
            'clip_ploss_coef_rate': self.clip_ploss_coef_rate,
            'clip_vloss_coef': self.clip_vloss_coef,
            'clip_advantage_lower_quantile': self.clip_advantage_lower_quantile,
            'clip_advantage_upper_quantile': self.clip_advantage_upper_quantile,
            'norm_adv': self.norm_adv
        })
        
        return config
    

    @classmethod
    def from_config(cls, config):
        gamma_denoising = config.pop("gamma_denoising", None)
        clip_ploss_coef = config.pop("clip_ploss_coef", None)
        clip_ploss_coef_base = config.pop("clip_ploss_coef_base", None)
        clip_ploss_coef_rate = config.pop("clip_ploss_coef_rate", None)
        clip_vloss_coef = config.pop("clip_vloss_coef", None)
        clip_advantage_lower_quantile = config.pop("clip_advantage_lower_quantile", None)
        clip_advantage_upper_quantile = config.pop("clip_advantage_upper_quantile", None)
        norm_adv = config.pop("norm_adv", None)

        parent_instance = super().from_config(config)


        print("parent_instance = ", parent_instance)


        return cls(gamma_denoising=gamma_denoising, 
                   clip_ploss_coef=clip_ploss_coef,
                   clip_ploss_coef_base=clip_ploss_coef_base,
                   clip_ploss_coef_rate=clip_ploss_coef_rate,
                   clip_vloss_coef=clip_vloss_coef,
                   clip_advantage_lower_quantile=clip_advantage_lower_quantile,
                   clip_advantage_upper_quantile=clip_advantage_upper_quantile,
                   norm_adv=norm_adv,
                   **parent_instance.get_config())





















    def estimate_entropy(self, actions, num_components=3):  # (batch, sample, dim)
        import numpy as np
        from sklearn.mixture import GaussianMixture
        total_entropy = []
        

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






    # def estimate_entropy(self, actions, num_components=3):
    #     import numpy as np
    #     from sklearn.mixture import GaussianMixture
    #     import tensorflow_probability as tfp

    #     total_entropy = []

    #     actions_np = actions.numpy()
    #     shape = actions_np.shape

    #     # reshaped_actions = tf.reshape(actions, (shape[0], -1))

    #     actions_np = actions_np.reshape(shape[0], -1)

    #     # for action in actions_np:
    #     action = actions_np
    #     means_list = []
    #     covariances_list = []

    #     gmm = GaussianMixture(n_components=num_components, covariance_type='full')
    #     gmm.fit(action)
    #     weights = gmm.weights_

    #     entropies = []
    #     for i in range(gmm.n_components):
    #         means_list.append(gmm.means_[i])
    #         covariances_list.append(gmm.covariances_[i])

    #         cov_matrix = gmm.covariances_[i]
    #         d = cov_matrix.shape[0]
    #         entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(cov_matrix)[1]
    #         entropies.append(entropy)

    #     entropy = -np.sum(weights * np.log(weights)) + np.sum(weights * np.array(entropies))
    #     total_entropy.append(entropy)

    #     # log_prob = torch_zeros_like(reshaped_actions)

    #     # print("reshaped_actions.shape = ", reshaped_actions.shape)
    #     # print("log_prob.shape = ", log_prob.shape)
    #     # print("weights = ", weights)

    #     # for k in range(num_components):
    #     #     tf_means = tf.convert_to_tensor(means_list[k], dtype=tf.float32)
    #     #     tf_covariances = tf.convert_to_tensor(covariances_list[k], dtype=tf.float32)
    #     #     # dist = tfp.distributions.MultivariateNormalFullCovariance(loc = tf_means, covariance_matrix = tf_covariances)

    #     #     print("tf_means.shape = ", tf_means.shape)
    #     #     print("tf_covariances.shape = ", tf_covariances.shape)

    #     #     dist = tfp.distributions.MultivariateNormalTriL(
    #     #         loc=tf_means,
    #     #         scale_tril=tf.linalg.cholesky(tf_covariances)
    #     #     )

    #     #     cur_log_probs = dist.log_prob(reshaped_actions)
    #     #     log_prob += weights[k] * cur_log_probs




    #     total_entropy_sum = sum(total_entropy)
    #     total_entropy_len = len(total_entropy)

    #     # log_prob = log_prob.reshape(shape)

    #     print("estimate_entropy(): total_entropy_sum = ", total_entropy_sum)
    #     # print("estimate_entropy(): log_prob = ", log_prob)
        


    #     return total_entropy_sum, total_entropy_len
    # # , log_prob






    def sac_get_logprobs_subsample(
        self,
        cond,
        chains_prev,
        chains_next,
        denoising_inds,
        # get_ent: bool = False,
        get_ent: bool = True,
        use_base_policy: bool = False,
    ):
        """
        Calculating the logprobs of random samples of denoised chains.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B, Ta, Da)
            entropy (if get_ent=True):  (B, Ta)
            denoising_indices: (B, )
        """

        print("diffusion_vpg.py: VPGDiffusion.get_logprobs_subsample()")



        # print("diffusion_ppo.py: VPGDiffusion.get_logprobs_subsample(): self.model.actor_ft.trainable_variables = ")
        # for var in self.actor_ft.trainable_variables:
        #     print(f"Variable: {var.name}, Trainable: {var.trainable}")


        # Sample t for batch dim, keep it 1-dim
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = torch_arange(
                start=self.ft_denoising_steps - 1,
                end=-1,
                step=-1,
                device=self.device,
            )
            # 4,3,2,1,0,4,3,2,1,0,...,4,3,2,1,0

        t_all = tf.gather( t_single, denoising_inds )


        # t_all = t_single[denoising_inds]


        if OUTPUT_POSITIONS:
            print("diffusion_vpg.py: VPGDiffusion.get_logprobs_subsample(): 1")

        if self.use_ddim:
            ddim_indices_single = torch_arange(
                start=self.ddim_steps - self.ft_denoising_steps,
                end=self.ddim_steps,
                device=self.device,
            )  # only used for DDIM
            # ddim_indices = ddim_indices_single[denoising_inds]
            ddim_indices = tf.gather(ddim_indices_single, denoising_inds)
        else:
            ddim_indices = None

        # Forward pass with previous chains

        if OUTPUT_POSITIONS:
            print("VPGDiffusion: get_logprobs_subsample() before p_mean_var()")

        mean, logvar, _ = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=ddim_indices,
            use_base_policy=use_base_policy,
        )

        # std = torch_exp(0.5 * logvar)
        # std = torch_clip(std, self.min_logprob_denoising_std, tf.float32.max)

        # dist = Normal(next_mean, std)

        # # Get logprobs with gaussian
        # log_prob = dist.log_prob(chains_next)

        min_sampling_denoising_std = self.get_min_sampling_denoising_std()


        std = torch_exp(0.5 * logvar)

        # Determine noise level
        if self.use_ddim:
            # if deterministic:
            #     std = torch_zeros_like(std)
            # else:
            std = torch_clip(std, min_sampling_denoising_std, tf.float32.max)
        else:
            # if deterministic and t == 0:
            #     std = torch_zeros_like(std)
            # elif deterministic:  # still keep the original noise
            #     std = torch_clip(std, 1e-3, tf.float32.max)
            # else:  # use higher minimum noise
            std = torch_clip(std, min_sampling_denoising_std, tf.float32.max)
        

        if DEBUG or NP_RANDOM:
            noise = tf.convert_to_tensor( np.random.randn( *(chains_prev.numpy().shape) ), dtype=tf.float32 )
        else:
            noise = torch_randn_like(chains_prev)


        if OUTPUT_VARIABLES:
            print("VPGDiffusion: call(): noise = ", noise)


        # temp_noise_variable = tf.Variable(temp_noise)

        # print("temp_noise = ", temp_noise)
        # print("temp_noise_variable = ", temp_noise_variable)

        # print("self.randn_clip_value = ", self.randn_clip_value)

        noise = torch_clamp(
            noise, -self.randn_clip_value, self.randn_clip_value
        )

        # noise = temp_noise_variable

        # print("std = ", std)
        # print("noise = ", noise)


        chains_prev_next = mean + std * noise

        # # clamp action at final step
        # if self.final_action_clip_value is not None and i == len(t_all) - 1:
        #     chains_prev_next = torch_clamp(chains_prev_next, -self.final_action_clip_value, self.final_action_clip_value)



        # total_entropy_sum, total_entropy_len, log_prob = self.estimate_entropy(chains_prev_next)

        import time

        time1 = time.time()

        if self.step_count % self.delay_alpha_update == 0:
            total_entropy_sum, total_entropy_len = self.estimate_entropy(chains_prev_next.numpy())
            entropy = total_entropy_sum / total_entropy_len
        else:
            entropy = self.entropy        

        time2 = time.time()
        elapsed_time = time2 - time1
        print(f"Elapsed time: single estimate_entropy {elapsed_time:.4f} seconds")

        self.step_count += 1



        dist = Normal(mean, std)

        # Get logprobs with gaussian
        log_prob = torch_tensor_float( dist.log_prob(chains_next) )


        if get_ent:
            # return log_prob, eta
            return log_prob, entropy

        return log_prob












    def loss_ori(
        self,
        obs,
        chains_prev,
        chains_next,
        denoising_inds,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
    ):
        """
        PPO loss

        obs: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        chains: (B, K+1, Ta, Da)
        returns: (B, )
        values: (B, )
        advantages: (B,)
        oldlogprobs: (B, K, Ta, Da)
        use_bc_loss: whether to add BC regularization loss
        reward_horizon: action horizon that backpropagates gradient
        """

        print("diffusion_ppo.py: PPODiffusion.loss()")

        # Get new logprobs for denoising steps from T-1 to 0 - entropy is fixed fod diffusion
        # print("diffusion_ppo.py: PPODiffusion.loss(): 1")



        newlogprobs, entropy = self.sac_get_logprobs_subsample(
            obs,
            chains_prev,
            chains_next,
            denoising_inds,
            get_ent=True,
        )

        self.entropy = entropy

        if self.entropy > self.target_entropy / 2:
            self.entropy = self.target_entropy / 2
        elif self.entropy < self.target_entropy * 2:
            self.entropy = self.target_entropy * 2

        entropy_loss = -torch_mean( torch_log(self.alpha) * ( -self.entropy + self.target_entropy ) )

        # print("diffusion_ppo.py: PPODiffusion.loss(): 2")

        # entropy_loss = -torch_mean(eta)



        newlogprobs = torch_clamp(newlogprobs, -5, 2)
        oldlogprobs = torch_clamp(oldlogprobs, -5, 2)

        # only backpropagate through the earlier steps (e.g., ones actually executed in the environment)
        newlogprobs = newlogprobs[:, :reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :reward_horizon, :]

        # Get the logprobs - batch over B and denoising steps
        newlogprobs = torch_mean(newlogprobs, dim=(-1, -2))
        oldlogprobs = torch_mean(oldlogprobs, dim=(-1, -2))
        newlogprobs = torch_tensor_view(newlogprobs, [-1])
        oldlogprobs = torch_tensor_view(oldlogprobs, [-1])

        # print("diffusion_ppo.py: PPODiffusion.loss(): 3")

        bc_loss = 0

        print("PPODiffusion: loss_ori(): use_bc_loss = ", use_bc_loss)

        if use_bc_loss:
            # See Eqn. 2 of https://arxiv.org/pdf/2403.03949.pdf
            # Give a reward for maximizing probability of teacher policy's action with current policy.
            # Actions are chosen along trajectory induced by current policy.

            # Get counterfactual teacher actions
            samples = self.call(
                cond=obs,
                deterministic=False,
                return_chain=True,
                use_base_policy=True,
            )

            # Get logprobs of teacher actions under this policy
            bc_logprobs = self.get_logprobs(
                obs,
                samples.chains,
                get_ent=False,
                use_base_policy=False,
            )


            bc_logprobs = torch_clamp(bc_logprobs, clip_value_min=-5, clip_value_max=2)
            bc_logprobs = torch_mean(bc_logprobs, axis=(-1, -2))
            bc_logprobs = torch_tensor_view(bc_logprobs, [-1])
            bc_loss = -torch_mean(bc_logprobs)

        # print("diffusion_ppo.py: PPODiffusion.loss(): 4")


        # normalize advantages
        if self.norm_adv:
            advantages = (advantages - torch_mean(advantages)) / (torch_std(advantages) + 1e-8)

        # Clip advantages by 5th and 95th percentile
        advantage_min = torch_quantile(advantages, self.clip_advantage_lower_quantile)
        advantage_max = torch_quantile(advantages, self.clip_advantage_upper_quantile)

        advantages = torch_clamp(advantages, advantage_min, advantage_max)


        # print("diffusion_ppo.py: PPODiffusion.loss(): 5")


        # print("type(self.ft_denoising_steps) = ", type(self.ft_denoising_steps))

        # denoising discount
        temp_discount = []
        for i in denoising_inds:
            square_num = (self.ft_denoising_steps - i - 1).numpy().item()
            # print("square_num = ", square_num)
            temp_discount.append(self.gamma_denoising ** square_num)
        # print("temp_discount = ", temp_discount)


        discount = torch_tensor(
            np.array(temp_discount)
        )

        # discount = torch_tensor(
        #     [
        #         self.gamma_denoising ** ( (self.ft_denoising_steps - i - 1).numpy().item() )
        #         for i in denoising_inds
        #     ]
        # )

        # .to(self.device)
        
        # print("advantages = ", advantages)
        # print("discount = ", discount)
        
        discount = tf.cast(discount, tf.float32)

        advantages *= discount

        # get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = torch_exp(logratio)

        # exponentially interpolate between the base and the current clipping value over denoising steps and repeat
        t = torch_tensor_float(denoising_inds) / (self.ft_denoising_steps - 1)

        # print("diffusion_ppo.py: PPODiffusion.loss(): 6")

        if self.ft_denoising_steps > 1:
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch_exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_ploss_coef = t

        # print("diffusion_ppo.py: PPODiffusion.loss(): 7")

        # get kl difference and whether value clipped
        # old_approx_kl: the approximate Kullback–Leibler divergence, measured by (-logratio).mean(), which corresponds to the k1 estimator in John Schulman’s blog post on approximating KL http://joschu.net/blog/kl-approx.html
        # approx_kl: better alternative to old_approx_kl measured by (logratio.exp() - 1) - logratio, which corresponds to the k3 estimator in approximating KL http://joschu.net/blog/kl-approx.html
        # old_approx_kl = (-logratio).mean()


        with torch_no_grad() as tape:
            approx_kl = torch_mean((ratio - 1) - logratio)
            clipfrac = torch_mean( torch_tensor_float( torch_abs(ratio - 1.0) > clip_ploss_coef ) )


        # print("advantages = ", advantages)
        # print("ratio = ", ratio)
        
        # Policy loss with clipping

        # pg_loss1 = -advantages * ratio
        pg_loss1 = -advantages * ratio + self.alpha * newlogprobs / 10
        pg_loss2 = -advantages * torch_clamp(ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef)

        # print("pg_loss1 = ", pg_loss1)
        # print("pg_loss2 = ", pg_loss2)
        # print("pg_loss1.shape = ", pg_loss1.shape)
        # print("pg_loss2.shape = ", pg_loss2.shape)
        
        pg_loss = torch_mean( torch_max(pg_loss1, pg_loss2) )

        # actor_loss = (self.alpha * log_prob - torch.min(q1_value, q2_value)).mean()
        # pg_loss = torch_mean(self.alpha * newlogprobs + torch_max(pg_loss1, pg_loss2))


        # Value loss optionally with clipping
        newvalues = self.critic(obs)
        newvalues = torch_tensor_view(newvalues, [-1])

        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch_clamp(
                newvalues - oldvalues, -self.clip_vloss_coef, self.clip_vloss_coef
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch_mean( torch_max(v_loss_unclipped, v_loss_clipped))

        else:
            v_loss = 0.5 * torch_mean( (newvalues - returns) ** 2 )

        # print("diffusion_ppo.py: PPODiffusion.loss(): 8")


        return (
            #pg_loss is actor loss
            pg_loss,
            # entropy loss is going to be the loss_temperature
            entropy_loss,
            # v_loss is critic loss
            v_loss,
            clipfrac,
            torch_tensor_item( approx_kl ),
            torch_tensor_item( torch_mean(ratio) ),
            bc_loss,
            self.entropy,
        )


















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

        # D3IL - avoid_m1/m2/m3，这几个都是avoiding-m5
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
        # next_q1_mean, next_q1_std, next_q2_mean, next_q2_std = critic(
        #     build_dict,
        #     param1,
        # )

        # self.loss_ori_build(actor, training=False, x_start = param1, cond=build_dict)

        # print("all_one_build_result next_q1_mean = ", sum(next_q1_mean))
        # print("all_one_build_result next_q1_std = ", sum(next_q1_mean))
        # print("all_one_build_result next_q2_mean = ", sum(next_q1_mean))
        # print("all_one_build_result next_q2_std = ", sum(next_q1_mean))





























