"""
Reward-weighted regression (RWR) for Gaussian policy.

"""

import tensorflow as tf

import logging
from model.common.gaussian import GaussianModel


from util.torch_to_tf import Normal

from util.torch_to_tf import torch_no_grad, torch_mean, torch_tensor_view

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
    def loss_ori(self, actions, obs, reward_weights):

        print("gaussian_rwr.py: RWR_Gaussian.loss()")

        B= obs.shape.as_list()[0]
        means, scales = self.network(obs)

        dist = Normal(means, scales)
        log_prob = dist.log_prob( torch_tensor_view(actions, [B, -1]) )
        log_prob = torch_mean(log_prob, dim = -1)

        log_prob = log_prob * reward_weights
        log_prob = -torch_mean(log_prob)
        return log_prob

    # override
    # @torch.no_grad()
    @tf.function
    def call(self, cond, deterministic=False, **kwargs):
        with torch_no_grad() as tape:

            print("gaussian_rwr.py: RWR_Gaussian.call()")

            actions = super().call(
                cond=cond,
                deterministic=deterministic,
            )
            return actions



















