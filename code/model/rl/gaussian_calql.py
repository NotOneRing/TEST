"""
Calibrated Conservative Q-Learning (CalQL) for Gaussian policy.

"""


import tensorflow as tf

from util.torch_to_tf import torch_min, torch_argmax, torch_arange, torch_tensor, torch_cat, torch_logsumexp, torch_tensor_view

from util.torch_to_tf import torch_prod, torch_max, torch_clamp, torch_mean, torch_tensor_view, torch_mse_loss, torch_repeat_interleave

from util.torch_to_tf import torch_no_grad

import logging
from copy import deepcopy
import numpy as np
import einops

from model.common.gaussian import GaussianModel

log = logging.getLogger(__name__)


class CalQL_Gaussian(GaussianModel):
    def __init__(
        self,
        actor,
        critic,
        network_path=None,
        cql_clip_diff_min=-np.inf,
        cql_clip_diff_max=np.inf,
        cql_min_q_weight=5.0,
        cql_n_actions=10,
        **kwargs,
    ):

        print("gaussian_calql.py: CalQL_Gaussian.__init__()")

        super().__init__(network=actor, network_path=None, **kwargs)
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.cql_min_q_weight = cql_min_q_weight
        self.cql_n_actions = cql_n_actions

        # initialize critic networks
        self.critic = critic
        self.target_critic = deepcopy(critic)
        
        # Load pretrained weights if specified
        if network_path is not None:
            print("network_path = ", network_path)
            print("self.critic = ", self.critic)


            checkpoint = tf.keras.models.load_model(network_path)
            self.actor.set_weights(checkpoint["actor_weights"])
            self.critic.set_weights(checkpoint["critic_weights"])
            log.info("Loaded actor and critic from %s", network_path)

        log.info(
            f"Number of network parameters: {np.sum([np.prod(v.shape) for v in self.actor.trainable_variables])}"
        )







    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        random_actions,
        rewards,
        returns,
        terminated,
        gamma,
    ):

        print("gaussian_calql.py: CalQL_Gaussian.loss_critic()")

        B = actions.shape[0]

        # Get initial TD loss
        q_data1, q_data2 = self.critic(obs, actions)

        # repeat for action samples
        with torch_no_grad() as tape:
            next_obs_repeated = {"state": torch_repeat_interleave( next_obs["state"],
                self.cql_n_actions, dim=0 )
            }

            # Get the next actions and logprobs
            next_actions, next_logprobs = self.call(
                next_obs_repeated,
                deterministic=False,
                get_logprob=True,
            )
            next_q1, next_q2 = self.target_critic(next_obs_repeated, next_actions)
            next_q = torch_min(next_q1, other=next_q2)

            # Reshape the next_q to match the number of samples
            next_q = torch_tensor_view(next_q, B, self.cql_n_actions)  # (B, n_sample)
            next_logprobs = torch_tensor_view( next_logprobs, B, self.cql_n_actions)  # (B, n_sample)

            # Get the max indices over the samples, and index into the next_q and next_log_probs
            max_idx = torch_argmax(next_q, dim=1)
            next_q = next_q[torch_arange(B), max_idx]
            next_logprobs = next_logprobs[torch_arange(B), max_idx]

            # Get the target Q values
            target_q = rewards + gamma * (1 - terminated) * next_q

        # TD loss
        td_loss_1 = torch_mse_loss(q_data1, target_q)
        td_loss_2 = torch_mse_loss(q_data2, target_q)

        # Get actions and logprobs
        log_rand_pi = 0.5 ** torch_prod(torch_tensor(random_actions.shape[-2:]))
        pi_actions, log_pi = self.call(
            obs,
            deterministic=False,
            reparameterize=False,
            get_logprob=True,
        )  # no gradient
        pi_next_actions, log_pi_next = self.call(
            next_obs,
            deterministic=False,
            reparameterize=False,
            get_logprob=True,
        )  # no gradient

        # Random action Q values
        n_random_actions = random_actions.shape[1]
        obs_sample_state = {
            "state": torch_repeat_interleave( obs["state"], n_random_actions, dim=0)
        }
        random_actions = einops.rearrange(random_actions, "B N H A -> (B N) H A")

        # Get the random action Q-values
        q_rand_1, q_rand_2 = self.critic(obs_sample_state, random_actions)
        q_rand_1 = q_rand_1 - log_rand_pi
        q_rand_2 = q_rand_2 - log_rand_pi

        # Reshape the random action Q values to match the number of samples
        q_rand_1 = torch_tensor_view(q_rand_1, B, n_random_actions)  # (n_sample, B)
        q_rand_2 = torch_tensor_view(q_rand_2, B, n_random_actions)

        # Policy action Q values
        q_pi_1, q_pi_2 = self.critic(obs, pi_actions)
        q_pi_next_1, q_pi_next_2 = self.critic(next_obs, pi_next_actions)

        # Ensure calibration w.r.t. value function estimate
        q_pi_1 = torch_max(q_pi_1, returns)[:, None]  # (B, 1)
        q_pi_2 = torch_max(q_pi_2, returns)[:, None]  # (B, 1)
        q_pi_next_1 = torch_max(q_pi_next_1, returns)[:, None]  # (B, 1)
        q_pi_next_2 = torch_max(q_pi_next_2, returns)[:, None]  # (B, 1)

        # cql_importance_sample
        q_pi_1 = q_pi_1 - log_pi
        q_pi_2 = q_pi_2 - log_pi
        q_pi_next_1 = q_pi_next_1 - log_pi_next
        q_pi_next_2 = q_pi_next_2 - log_pi_next
        cat_q_1 = torch_cat([q_rand_1, q_pi_1, q_pi_next_1], dim=-1)  # (B, num_samples+1)
        cql_qf1_ood = torch_logsumexp(cat_q_1, dim=-1)  # max over num_samples
        cat_q_2 = torch_cat([q_rand_2, q_pi_2, q_pi_next_2], dim=-1)  # (B, num_samples+1)
        cql_qf2_ood = torch_logsumexp(cat_q_2, dim=-1)  # sum over num_samples

        # skip cal_lagrange since the paper shows cql_target_action_gap not used in kitchen

        # Subtract the log likelihood of the data
        cql_qf1_diff = torch_mean( torch_clamp(
            cql_qf1_ood - q_data1,
            min=self.cql_clip_diff_min,
            max=self.cql_clip_diff_max,
        )
        )
        cql_qf2_diff = torch_mean(
        torch_clamp(
            cql_qf2_ood - q_data2,
            min=self.cql_clip_diff_min,
            max=self.cql_clip_diff_max,
        )
        )
        cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
        cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight

        # Sum the two losses
        critic_loss = td_loss_1 + td_loss_2 + cql_min_qf1_loss + cql_min_qf2_loss
        return critic_loss






    def loss_actor(self, obs, alpha):

        print("gaussian_calql.py: CalQL_Gaussian.loss_actor()")

        action, logprob = self.call(
            obs,
            deterministic=False,
            reparameterize=True,
            get_logprob=True,
        )
        q1, q2 = self.critic(obs, action)
        actor_loss = -torch_min(q1, other=q2) + alpha * logprob
        return torch_mean(actor_loss)
    





    
    def loss_temperature(self, obs, alpha, target_entropy):

        print("gaussian_calql.py: CalQL_Gaussian.loss_temperature()")

        with torch_no_grad() as tape:
            _, logprob = self.call(
                obs,
                deterministic=False,
                get_logprob=True,
            )
        loss_alpha = -torch_mean(alpha * (logprob + target_entropy))
        return loss_alpha












    def update_target_critic(self, tau):
        print("gaussian_calql.py: CalQL_Gaussian.update_target_critic()")

        for target_param, param in zip(
            self.target_critic.trainable_variables, self.critic.trainable_variables
        ):
            updated_value = tau * param + (1 - tau) * target_param
            target_param.assign(updated_value)


















