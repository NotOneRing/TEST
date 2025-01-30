"""
Gaussian policy parameterization.

"""


import tensorflow as tf

import numpy as np

import logging

log = logging.getLogger(__name__)

from util.torch_to_tf import Normal, torch_ones_like, torch_clamp, torch_mean, torch_tensor_view,\
torch_log, torch_tanh, torch_sum



class GaussianModel(tf.keras.Model):

    def __init__(
        self,
        network,
        horizon_steps,
        network_path=None,
        device="cuda:0",
        randn_clip_value=10,
        tanh_output=False,
    ):

        print("gaussian.py: GaussianModel.__init__()")

        super().__init__()
        # self.device = device
        self.network = network
        # .to(device)

        if network_path is not None:
            checkpoint = tf.train.Checkpoint(model=self)
            checkpoint.restore(network_path).expect_partial()
            log.info("Loaded actor from %s", network_path)


        # Log number of parameters in the network
        num_params = sum(np.prod(var.shape) for var in self.network.trainable_variables)
        log.info(f"Number of network parameters: {num_params}")


        self.horizon_steps = horizon_steps

        # Clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Whether to apply tanh to the **sampled** action --- used in SAC
        self.tanh_output = tanh_output



    def loss(
        self,
        true_action,
        cond,
        ent_coef,
    ):
        """no squashing"""

        print("gaussian.py: GaussianModel.loss()")

        B = len(true_action)
        dist = self.forward_train(cond, deterministic=False)
        # true_action = tf.reshape(true_action, (B, -1))  # Flatten actions to shape [B, action_dim]
        true_action = torch_tensor_view(true_action, (B, -1))  # Flatten actions to shape [B, action_dim]
        log_prob = dist.log_prob(true_action)
        entropy = torch_mean(dist.entropy())
        loss = -torch_mean(log_prob) - entropy * ent_coef
        return loss, {"entropy": entropy}



    def forward_train(
        self,
        cond,
        deterministic=False,
        network_override=None,
    ):
        """
        Calls the MLP to compute the mean, scale, and logits of the GMM. Returns the torch.Distribution object.
        """

        print("gaussian.py: GaussianModel.forward_train()")

        if network_override is not None:
            means, scales = network_override(cond)
        else:
            means, scales = self.network(cond)
        if deterministic:
            # low-noise for all Gaussian dists
            scales = torch_ones_like(means) * 1e-4

        # dist = tfp.distributions.Normal(loc=means, scale=scales)
        dist = Normal(means, scales)

        return dist

    def call(
        self,
        cond,
        deterministic=False,
        network_override=None,
        reparameterize=False,
        get_logprob=False,
    ):

        print("gaussian.py: GaussianModel.call()")

        B = len(cond["state"]) if "state" in cond else len(cond["rgb"])
        T = self.horizon_steps
        dist = self.forward_train(
            cond,
            deterministic=deterministic,
            network_override=network_override,
        )

        if reparameterize:
            assert "reparameterize is not implemented right now"
            # sampled_action = dist.rsample()  # reparameterized sample
        else:
            sampled_action = dist.sample()  # standard sample


        # Clipping the sampled action (similar to PyTorch clamp_)
        sampled_action = torch_clamp(sampled_action, dist.loc - self.randn_clip_value * dist.scale,
                                          dist.loc + self.randn_clip_value * dist.scale)

        if get_logprob:
            log_prob = dist.log_prob(sampled_action)

            # For SAC/RLPD, squash mean after sampling here instead of right after model output as in PPO
            if self.tanh_output:
                sampled_action = torch_tanh(sampled_action)
                log_prob -= torch_log(1 - tf.square(sampled_action) + 1e-6)

            return torch_tensor_view(sampled_action, [B, T, -1]), torch_sum(log_prob, axis=1)
        else:
            if self.tanh_output:
                sampled_action = torch_tanh(sampled_action)
            return torch_tensor_view(sampled_action, (B, T, -1))

            





