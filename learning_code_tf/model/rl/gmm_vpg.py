import torch
import logging
from model.common.gmm import GMMModel


class VPG_GMM(GMMModel):
    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):

        print("gmm_vpg.py: VPG_GMM.__init__()", flush = True)

        super().__init__(network=actor, **kwargs)

        # Re-name network to actor
        self.actor_ft = actor

        # Value function for obs - simple MLP
        self.critic = critic.to(self.device)

    # ---------- Sampling ----------#

    @torch.no_grad()
    def forward(self, cond, deterministic=False):

        print("gmm_vpg.py: VPG_GMM.forward()", flush = True)

        return super().forward(
            cond=cond,
            deterministic=deterministic,
        )

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        actions,
    ):

        print("gmm_vpg.py: VPG_GMM.get_logprobs()", flush = True)

        B = len(actions)
        dist, entropy, std = self.forward_train(
            cond,
            deterministic=False,
        )
        log_prob = dist.log_prob(actions.view(B, -1))
        return log_prob, entropy, std

    def loss(self, obs, chains, reward):

        print("gmm_vpg.py: VPG_GMM.loss()", flush = True)

        raise NotImplementedError
