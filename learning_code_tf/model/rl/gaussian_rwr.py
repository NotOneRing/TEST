"""
Reward-weighted regression (RWR) for Gaussian policy.

"""

import torch
import logging
from model.common.gaussian import GaussianModel
import torch.distributions as D

log = logging.getLogger(__name__)


class RWR_Gaussian(GaussianModel):

    def __init__(
        self,
        actor,
        **kwargs,
    ):

        print("gaussian_rwr.py: RWR_Gaussian.__init__()")

        super().__init__(network=actor, **kwargs)

        # assign actor
        self.actor = self.network

    # override
    def loss(self, actions, obs, reward_weights):

        print("gaussian_rwr.py: RWR_Gaussian.loss()")

        B = len(obs)
        means, scales = self.network(obs)

        dist = D.Normal(loc=means, scale=scales)
        log_prob = dist.log_prob(actions.view(B, -1)).mean(-1)
        log_prob = log_prob * reward_weights
        log_prob = -log_prob.mean()
        return log_prob

    # override
    @torch.no_grad()
    def forward(self, cond, deterministic=False, **kwargs):

        print("gaussian_rwr.py: RWR_Gaussian.forward()")

        actions = super().forward(
            cond=cond,
            deterministic=deterministic,
        )
        return actions
