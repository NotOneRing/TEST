"""
Advantage-weighted regression (AWR) for Gaussian policy.

"""

import tensorflow as tf
import logging
from model.rl.gaussian_rwr import RWR_Gaussian

log = logging.getLogger(__name__)


class AWR_Gaussian(RWR_Gaussian):

    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):

        print("gaussian_awr.py: AWR_Gaussian.__init__()")

        super().__init__(actor=actor, **kwargs)

        self.critic = critic

    def loss_critic(self, obs, advantages):

        print("gaussian_awr.py: AWR_Gaussian.loss_critic()")

        # get advantage
        adv = self.critic(obs)

        # Update critic
        loss_critic = tf.reduce_mean( tf.square(adv - advantages) )

        return loss_critic


