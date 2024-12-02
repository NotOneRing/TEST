"""
Advantage-weighted regression (AWR) for diffusion policy.

"""

import logging

import tensorflow as tf

log = logging.getLogger(__name__)

from model.diffusion.diffusion_rwr import RWRDiffusion


class AWRDiffusion(RWRDiffusion):

    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):

        print("diffusion_awr.py: AWRDiffusion.__init__()")

        super().__init__(network=actor, **kwargs)
        self.critic = critic
        
        # assign actor
        self.actor = self.network

    def loss_critic(self, obs, advantages):

        print("diffusion_awr.py: AWRDiffusion.loss_critic()")

        # get advantage
        adv = self.critic(obs)

        # Update critic
        loss_critic = tf.reduce_mean(tf.square(adv - advantages))

        return loss_critic
