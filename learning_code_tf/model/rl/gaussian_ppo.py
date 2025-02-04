"""
PPO for Gaussian policy.

To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""

from typing import Optional

import tensorflow as tf


from model.rl.gaussian_vpg import VPG_Gaussian

from util.torch_to_tf import torch_no_grad, torch_nanmean, torch_mean, torch_tensor_float, torch_abs, torch_tensor_item, \
torch_clamp, torch_exp, torch_std, torch_max, torch_tensor_view


class PPO_Gaussian(VPG_Gaussian):

    def __init__(
        self,
        clip_ploss_coef: float,
        clip_vloss_coef: Optional[float] = None,
        norm_adv: Optional[bool] = True,
        **kwargs,
    ):

        print("gaussian_ppo.py: PPO_Gaussian.__init__()")

        super().__init__(**kwargs)

        # Whether to normalize advantages within batch
        self.norm_adv = norm_adv

        # Clipping value for policy loss
        self.clip_ploss_coef = clip_ploss_coef

        # Clipping value for value loss
        self.clip_vloss_coef = clip_vloss_coef

    def loss(
        self,
        obs,
        actions,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
    ):
        """
        PPO loss

        obs: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        actions: (B, Ta, Da)
        returns: (B, )
        values: (B, )
        advantages: (B,)
        oldlogprobs: (B, )
        """

        print("gaussian_ppo.py: PPO_Gaussian.loss()")

        # Get new log probabilities and entropy
        newlogprobs, entropy, std = self.get_logprobs(obs, actions)
        newlogprobs = torch_clamp(newlogprobs, -5.0, 2.0)
        oldlogprobs = torch_clamp(oldlogprobs, -5.0, 2.0)
        entropy_loss = -entropy
        
        bc_loss = 0.0
        if use_bc_loss:
            # See Eqn. 2 of https://arxiv.org/pdf/2403.03949.pdf
            # Give a reward for maximizing probability of teacher policy's action with current policy.
            # Actions are chosen along trajectory induced by current policy.

            # Get counterfactual teacher actions
            samples = self.call(
                cond=obs,
                deterministic=False,
                use_base_policy=True,
            )

            # Get logprobs of teacher actions under this policy
            bc_logprobs, _, _ = self.get_logprobs(obs, samples, use_base_policy=False)

            bc_logprobs = torch_clamp(bc_logprobs, -5.0, 2.0)
            bc_loss = -torch_mean(bc_logprobs)

        # get ratio
        logratio = newlogprobs - oldlogprobs

        ratio = torch_exp(logratio)

        # # get kl difference and whether value clipped
        # with torch_no_grad() as tape:
        #     approx_kl = tf.reduce_mean(
        #         tf.boolean_mask((ratio - 1) - logratio, ~tf.is_nan((ratio - 1) - logratio))
        #     )

        #     clipfrac = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > self.clip_ploss_coef, tf.float32))


        # get kl difference and whether value clipped
        with torch_no_grad() as tape:
            approx_kl = torch_nanmean( (ratio - 1) - logratio )
            clipfrac = (
                torch_tensor_item( torch_mean( torch_tensor_float( torch_abs(ratio - 1.0) > self.clip_ploss_coef) ) )
            )

        # Normalize advantages
        if self.norm_adv:
            advantages = (advantages - torch_mean(advantages)) / ( torch_std(advantages) + 1e-8 )


        # Policy loss with clipping
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch_clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch_mean( torch_max(pg_loss1, pg_loss2) )


        # Value loss optionally with clipping
        newvalues = self.critic(obs)
        newvalues = torch_tensor_view(newvalues, [-1])

        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch_clamp(newvalues - oldvalues, -self.clip_vloss_coef, self.clip_vloss_coef)
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss = 0.5 * torch_mean( torch_clamp(v_loss_unclipped, v_loss_clipped) )
        else:
            v_loss = 0.5 * torch_mean((newvalues - returns) ** 2)

        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            torch_tensor_item( approx_kl ),
            torch_tensor_item( torch_mean(ratio) ),
            bc_loss,
            torch_tensor_item( std ),
        )











