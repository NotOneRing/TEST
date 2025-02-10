"""
Soft Actor Critic (SAC) with Gaussian policy.

"""

import logging
from copy import deepcopy


from model.common.gaussian import GaussianModel

log = logging.getLogger(__name__)

from util.torch_to_tf import torch_mse_loss, torch_min, torch_mean

from util.torch_to_tf import torch_no_grad, torch_mean

import tensorflow as tf

class SAC_Gaussian(GaussianModel):
    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):

        print("gaussian_sac.py: SAC_Gaussian.__init__()")

        super().__init__(network=actor, **kwargs)

        # initialize doubel critic networks
        self.critic = critic
        # .to(self.device)

        # initialize double target networks
        self.target_critic = deepcopy(self.critic)
        # .to(self.device)

        self.actor = self.network

        

    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        rewards,
        terminated,
        gamma,
        alpha,
    ):

        print("gaussian_sac.py: SAC_Gaussian.loss_critic()")

        # with torch.no_grad():
        with torch_no_grad() as tape:
            next_actions, next_logprobs = self.call(
                cond=next_obs,
                deterministic=False,
                get_logprob=True,
            )
            next_q1, next_q2 = self.target_critic(
                next_obs,
                next_actions,
            )
            next_q = torch_min(next_q1, other=next_q2) - alpha * next_logprobs

            # target value
            target_q = rewards + gamma * next_q * (1 - terminated)

        current_q1, current_q2 = self.critic(obs, actions)
        loss_critic = torch_mse_loss(current_q1, target_q) + torch_mse_loss(
            current_q2, target_q
        )
        return loss_critic

    def loss_actor(self, obs, alpha):

        print("gaussian_sac.py: SAC_Gaussian.loss_actor()")

        action, logprob = self.call(
            obs,
            deterministic=False,
            reparameterize=True,
            get_logprob=True,
        )
        current_q1, current_q2 = self.critic(obs, action)
        loss_actor = -torch_min(current_q1, current_q2) + alpha * logprob
        return torch_mean(loss_actor)
    
    def loss_temperature(self, obs, alpha, target_entropy):

        print("gaussian_sac.py: SAC_Gaussian.loss_temperature()")

        # with torch.no_grad():
        with torch_no_grad() as tape:
            _, logprob = self.call(
                obs,
                deterministic=False,
                get_logprob=True,
            )

        loss_alpha = -torch_mean(alpha * (logprob + target_entropy))
        return loss_alpha

    def update_target_critic(self, tau):

        print("gaussian_sac.py: SAC_Gaussian.update_target_critic()")

        critic_variables = self.critic.trainable_variables
        target_critic_variables = self.target_critic.trainable_variables

        for target_param, source_param in zip(target_critic_variables, critic_variables):
            target_param.assign(target_param * (1.0 - tau) + source_param * tau)











