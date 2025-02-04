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

from util.torch_to_tf import torch_quantile
# import tensorflow_probability as tfp

from util.torch_to_tf import torch_no_grad, torch_tensor_item

from util.torch_to_tf import torch_clamp, torch_exp, torch_mean, torch_max, torch_tensor_float, torch_abs, torch_tensor, torch_std, torch_tensor_view


import numpy as np


log = logging.getLogger(__name__)
from model.diffusion.diffusion_vpg import VPGDiffusion


from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, OUTPUT_POSITIONS, OUTPUT_FUNCTION_HEADER


class PPODiffusion(VPGDiffusion):
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


        newlogprobs, eta = self.get_logprobs_subsample(
            obs,
            chains_prev,
            chains_next,
            denoising_inds,
            get_ent=True,
        )

        # print("diffusion_ppo.py: PPODiffusion.loss(): 2")

        entropy_loss = -torch_mean(eta)
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
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch_clamp(ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef)

        # print("pg_loss1 = ", pg_loss1)
        # print("pg_loss2 = ", pg_loss2)
        # print("pg_loss1.shape = ", pg_loss1.shape)
        # print("pg_loss2.shape = ", pg_loss2.shape)
        
        pg_loss = torch_mean( torch_max(pg_loss1, pg_loss2) )


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
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            torch_tensor_item( approx_kl ),
            torch_tensor_item( torch_mean(ratio) ),
            bc_loss,
            torch_tensor_item( torch_mean(eta) ),
        )


























