"""
Advantage-weighted regression (AWR) for Gaussian policy.

"""

import tensorflow as tf
import logging
from model.rl.gaussian_rwr import RWR_Gaussian

log = logging.getLogger(__name__)

from util.torch_to_tf import *

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
        loss_critic = torch_mean( (adv - advantages)**2 )

        return loss_critic


