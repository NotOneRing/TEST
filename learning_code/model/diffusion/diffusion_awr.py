"""
Advantage-weighted regression (AWR) for diffusion policy.

"""

import logging
import torch

log = logging.getLogger(__name__)

from model.diffusion.diffusion_rwr import RWRDiffusion


class AWRDiffusion(RWRDiffusion):

    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):

        print("diffusion_awr.py: AWRDiffusion.__init__()", flush = True)

        super().__init__(network=actor, **kwargs)
        self.critic = critic.to(self.device)

        # assign actor
        self.actor = self.network

    def loss_critic(self, obs, advantages):

        print("diffusion_awr.py: AWRDiffusion.loss_critic()", flush = True)

        # get advantage
        adv = self.critic(obs)

        # Update critic
        loss_critic = torch.mean((adv - advantages) ** 2)
        return loss_critic