"""
Diffusion policy gradient with exact likelihood estimation.

Based on score_sde_pytorch https://github.com/yang-song/score_sde_pytorch

To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

"""

import tensorflow as tf

import logging

log = logging.getLogger(__name__)
from .diffusion_vpg import VPGDiffusion
from .exact_likelihood import get_likelihood_fn

from util.torch_to_tf import torch_no_grad

class PPOExactDiffusion(VPGDiffusion):

    def __init__(
        self,
        sde,
        clip_ploss_coef,
        clip_vloss_coef=None,
        norm_adv=True,
        sde_hutchinson_type="Rademacher",
        sde_rtol=1e-4,
        sde_atol=1e-4,
        sde_eps=1e-4,
        sde_step_size=1e-3,
        sde_method="RK23",
        sde_continuous=False,
        sde_probability_flow=False,
        sde_num_epsilon=1,
        sde_min_beta=1e-2,
        **kwargs,
    ):

        print("diffusion_ppo_exact.py: PPOExactDiffusion.__init__()")
        
        super().__init__(**kwargs)
        self.sde = sde
        self.sde.set_betas(
            self.betas,
            sde_min_beta,
        )
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_vloss_coef = clip_vloss_coef
        self.norm_adv = norm_adv

        # set up likelihood function
        self.likelihood_fn = get_likelihood_fn(
            sde,
            hutchinson_type=sde_hutchinson_type,
            rtol=sde_rtol,
            atol=sde_atol,
            eps=sde_eps,
            step_size=sde_step_size,
            method=sde_method,
            continuous=sde_continuous,
            probability_flow=sde_probability_flow,
            predict_epsilon=self.predict_epsilon,
            num_epsilon=sde_num_epsilon,
        )

    def get_exact_logprobs(self, cond, samples):
        """Use torchdiffeq

        samples: (B x Ta x Da)
        """

        print("diffusion_ppo_exact.py: PPOExactDiffusion.get_exact_logprobs()")

        return self.likelihood_fn(
            self.actor,
            self.actor_ft,
            samples,
            self.denoising_steps,
            self.ft_denoising_steps,
            cond=cond,
        )

    def loss(
        self,
        obs,
        samples,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        **kwargs,
    ):
        """
        PPO loss

        obs: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        samples: (B, Ta, Da)
        returns: (B, )
        values: (B, )
        advantages: (B,)
        oldlogprobs: (B, )
        """

        print("diffusion_ppo_exact.py: PPOExactDiffusion.loss()")

        # Get new logprobs for final x
        newlogprobs = self.get_exact_logprobs(obs, samples)
        newlogprobs = tf.clip_by_value(newlogprobs, clip_value_min=-5, clip_value_max=2)
        oldlogprobs = tf.clip_by_value(oldlogprobs, clip_value_min=-5, clip_value_max=2)


        bc_loss = 0
        if use_bc_loss:
            raise NotImplementedError

        # get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = tf.exp(logratio)


        # get kl difference and whether value clipped
        # old_approx_kl: the approximate Kullback–Leibler divergence, measured by (-logratio).mean(), which corresponds to the k1 estimator in John Schulman’s blog post on approximating KL http://joschu.net/blog/kl-approx.html
        # approx_kl: better alternative to old_approx_kl measured by (logratio.exp() - 1) - logratio, which corresponds to the k3 estimator in approximating KL http://joschu.net/blog/kl-approx.html
        # old_approx_kl = (-logratio).mean()
        with torch_no_grad() as tape:
            approx_kl = tf.reduce_mean((ratio - 1) - logratio)
            clipfrac = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > self.clip_ploss_coef, tf.float32))


        # Normalize advantages
        if self.norm_adv:
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.reduce_std(advantages) + 1e-8)

        # Policy loss with clipping
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * tf.clip_by_value(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))


        # Value loss optionally with clipping
        newvalues = self.critic(obs)
        newvalues = tf.reshape(newvalues, [-1])
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = tf.square(newvalues - returns)
            v_clipped = oldvalues + tf.clip_by_value(newvalues - oldvalues, -self.clip_vloss_coef, self.clip_vloss_coef)
            v_loss_clipped = tf.square(v_clipped - returns)
            v_loss_max = tf.maximum(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * tf.reduce_mean(v_loss_max)
        else:
            v_loss = 0.5 * tf.reduce_mean(tf.square(newvalues - returns))


        # entropy is maximized - only effective if residual is learned
        return (
            pg_loss,
            v_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            bc_loss,
        )
