"""
Policy gradient for Gaussian policy

"""

import tensorflow as tf

from copy import deepcopy
import logging
from model.common.gaussian import GaussianModel

from util.torch_to_tf import torch_no_grad


class VPG_Gaussian(GaussianModel):

    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):

        print("gaussian_vpg.py: VPG_Gaussian.__init__()")

        super().__init__(network=actor, **kwargs)

        # Value function for obs - simple MLP
        self.critic = critic
        # .to(self.device)

        # Re-name network to actor
        self.actor_ft = actor

        # Save a copy of original actor
        self.actor = deepcopy(actor)
        # for param in self.actor.parameters():
        #     param.requires_grad = False

        for layer in self.actor.layers:
            layer.trainable = False


    # ---------- Sampling ----------#

    # @torch.no_grad()
    @tf.function
    def call(
        self,
        cond,
        deterministic=False,
        use_base_policy=False,
    ):

        print("gaussian_vpg.py: VPG_Gaussian.call()")

        return super().call(
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

        print("gaussian_vpg.py: VPG_Gaussian.get_logprobs()")

        B = tf.shape(actions)[0]

        dist = self.forward_train(
            cond,
            deterministic=False,
            network_override=self.actor if use_base_policy else None,
        )

        # Compute log probability
        log_prob = dist.log_prob(tf.reshape(actions, [B, -1]))
        log_prob = tf.reduce_mean(log_prob, axis=-1)

        # Compute entropy and standard deviation
        entropy = tf.reduce_mean(dist.entropy())
        std = tf.reduce_mean(dist.stddev())

        return log_prob, entropy, std

    def loss(self, obs, actions, reward):

        print("gaussian_vpg.py: VPG_Gaussian.loss()")

        raise NotImplementedError
