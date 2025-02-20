"""
PPO for GMM policy.

To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""

from typing import Optional

import tensorflow as tf

from model.rl.gmm_vpg import VPG_GMM

from util.torch_to_tf import torch_clamp, torch_max, torch_nanmean, torch_std, torch_tensor_view

from util.torch_to_tf import torch_no_grad, torch_mean, torch_exp, torch_tensor_item, torch_tensor_float, torch_abs


class PPO_GMM(VPG_GMM):

    def __init__(
        self,
        clip_ploss_coef: float,
        clip_vloss_coef: Optional[float] = None,
        norm_adv: Optional[bool] = True,
        **kwargs,
    ):

        print("gmm_ppo.py: PPO_GMM.__init__()")

        super().__init__(**kwargs)

        # Whether to normalize advantages within batch
        self.norm_adv = norm_adv

        # Clipping value for policy loss
        self.clip_ploss_coef = clip_ploss_coef

        # Clipping value for value loss
        self.clip_vloss_coef = clip_vloss_coef

    def loss_ori(
        self,
        obs,
        actions,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        **kwargs,
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

        print("gmm_ppo.py: PPO_GMM.loss()")

        newlogprobs, entropy, std = self.get_logprobs(obs, actions)


        newlogprobs = torch_clamp(newlogprobs, min=-5, max=2)
        oldlogprobs = torch_clamp(oldlogprobs, min=-5, max=2)

        entropy_loss = -torch_mean( entropy )

        # get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = torch_exp( logratio )

        # get kl difference and whether value clipped
        # with torch.no_grad():
        # approx_kl = ((ratio - 1) - logratio).nanmean()

        with torch_no_grad() as tape:
            approx_kl = torch_nanmean((ratio - 1) - logratio)
            # clipfrac = tf.reduce_mean( tf.cast( tf.greater( tf.abs(ratio - 1.0), self.clip_ploss_coef ), tf.float32) )
            clipfrac = (
                torch_tensor_item( torch_mean( torch_tensor_float( ( torch_abs(ratio - 1.0)  > self.clip_ploss_coef ) ) ) )
            )


        advantages_std = torch_std(advantages)

        # normalize advantages
        if self.norm_adv:
            advantages = (advantages - torch_mean( advantages ) ) / ( advantages_std + 1e-8)

        # Policy loss with clipping
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch_clamp(
            ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef
        )

        pg_loss = torch_mean( torch_max(pg_loss1, pg_loss2) )

        # Value loss optionally with clipping
        newvalues = torch_tensor_view( self.critic(obs), [-1] )
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch_clamp(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch_max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * torch_mean( v_loss_max )
        else:
            v_loss = 0.5 * torch_mean((newvalues - returns) ** 2)
        bc_loss = 0
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            torch_tensor_item( approx_kl ),
            torch_tensor_item( torch_mean( ratio ) ),
            bc_loss,
            torch_tensor_item( std ),
        )














