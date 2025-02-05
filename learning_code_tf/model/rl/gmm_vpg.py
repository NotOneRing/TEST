import tensorflow as tf

import logging
from model.common.gmm import GMMModel

from util.torch_to_tf import torch_no_grad, torch_tensor_view


class VPG_GMM(GMMModel):
    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):

        print("gmm_vpg.py: VPG_GMM.__init__()")

        super().__init__(network=actor, **kwargs)

        # Re-name network to actor
        self.actor_ft = actor

        # Value function for obs - simple MLP
        self.critic = critic
        # .to(self.device)

    # ---------- Sampling ----------#

    # @torch.no_grad()
    @tf.function
    def call(self, cond, deterministic=False):
        print("gmm_vpg.py: VPG_GMM.call()")
        with torch_no_grad() as tape:
            return super().call(
                cond=cond,
                deterministic=deterministic,
            )

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        actions,
    ):

        print("gmm_vpg.py: VPG_GMM.get_logprobs()")

        # B = len(actions)
        B = actions.shape().as_list()[0]
        dist, entropy, std = self.forward_train(
            cond,
            deterministic=False,
        )
        log_prob = dist.log_prob( torch_tensor_view(actions, [B, -1]) )
        return log_prob, entropy, std

    def loss_ori(self, obs, chains, reward):

        print("gmm_vpg.py: VPG_GMM.loss()")

        raise NotImplementedError








