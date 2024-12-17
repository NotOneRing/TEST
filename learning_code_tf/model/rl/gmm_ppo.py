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

from util.torch_to_tf import torch_clamp, torch_max, torch_nanmean

from util.torch_to_tf import torch_no_grad


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

    def loss(
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


        newlogprobs = tf.clip_by_value(newlogprobs, min=-5, max=2)
        oldlogprobs = tf.clip_by_value(oldlogprobs, min=-5, max=2)

        entropy_loss = -tf.reduce_mean( entropy )

        # get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = tf.exp( logratio )

        # get kl difference and whether value clipped
        # with torch.no_grad():
        # approx_kl = ((ratio - 1) - logratio).nanmean()

        with torch_no_grad() as tape:
            approx_kl = torch_nanmean((ratio - 1) - logratio)
            clipfrac = tf.reduce_mean( tf.cast( tf.greater( tf.abs(ratio - 1.0), self.clip_ploss_coef ), tf.float32) )


        advantages_std = tf.reduce_std(advantages)

        # normalize advantages
        if self.norm_adv:
            advantages = (advantages - tf.reduce_mean( advantages ) ) / ( advantages_std + 1e-8)

        # Policy loss with clipping
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch_clamp(
            ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef
        )

        pg_loss = tf.reduce_mean( torch_max(pg_loss1, pg_loss2) )

        # Value loss optionally with clipping
        newvalues = tf.reshape( self.critic(obs), [-1] )
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch_clamp(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch_max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * tf.reduce_mean( v_loss_max )
        else:
            v_loss = 0.5 * tf.reduce_mean((newvalues - returns) ** 2)
        bc_loss = 0
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            int( approx_kl ),
            tf.reduce_mean( ratio ),
            bc_loss,
            int( std ),
        )
