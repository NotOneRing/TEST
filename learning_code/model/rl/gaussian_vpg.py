"""
Policy gradient for Gaussian policy

"""

import torch
from copy import deepcopy
import logging
from model.common.gaussian import GaussianModel


class VPG_Gaussian(GaussianModel):

    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):

        print("gaussian_vpg.py: VPG_Gaussian.__init__()", flush = True)

        super().__init__(network=actor, **kwargs)

        # Value function for obs - simple MLP
        self.critic = critic.to(self.device)

        # Re-name network to actor
        self.actor_ft = actor

        # Save a copy of original actor
        self.actor = deepcopy(actor)
        for param in self.actor.parameters():
            param.requires_grad = False

    # ---------- Sampling ----------#

    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        use_base_policy=False,
    ):

        print("gaussian_vpg.py: VPG_Gaussian.forward()", flush = True)

        return super().forward(
            cond=cond,
            deterministic=deterministic,
            network_override=self.actor if use_base_policy else None,
        )

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        actions,
        use_base_policy=False,
    ):

        print("gaussian_vpg.py: VPG_Gaussian.get_logprobs()", flush = True)

        B = len(actions)
        dist = self.forward_train(
            cond,
            deterministic=False,
            network_override=self.actor if use_base_policy else None,
        )
        log_prob = dist.log_prob(actions.view(B, -1))
        log_prob = log_prob.mean(-1)
        entropy = dist.entropy().mean()
        std = dist.scale.mean()
        return log_prob, entropy, std

    def loss(self, obs, actions, reward):

        print("gaussian_vpg.py: VPG_Gaussian.loss()", flush = True)

        raise NotImplementedError