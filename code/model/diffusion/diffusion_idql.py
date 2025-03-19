"""
Implicit diffusion Q-learning (IDQL) for diffusion policy.

"""

import logging

import einops
import copy

import tensorflow as tf

log = logging.getLogger(__name__)

from model.diffusion.diffusion_rwr import RWRDiffusion

from util.torch_to_tf import torch_gather, torch_no_grad, torch_where, torch_min, torch_reshape,\
torch_mean, torch_tensor_view, torch_randn_like, torch_mse_loss, torch_tensor_repeat, torch_argmax, \
torch_sum, torch_multinomial


def expectile_loss(diff, expectile=0.8):

    print("diffusion_idql.py: expectile_loss()")

    weight = torch_where(diff > 0, expectile, (1 - expectile))

    return weight * (diff**2)


class IDQLDiffusion(RWRDiffusion):

    def __init__(
        self,
        actor,
        critic_q,
        critic_v,
        **kwargs,
    ):

        print("diffusion_idql.py: IDQLDiffusion.__init__()")

        super().__init__(network=actor, **kwargs)
        self.critic_q = critic_q
        self.target_q = copy.deepcopy(critic_q)
        self.critic_v = critic_v

        self.actor = self.network

    # ---------- RL training ----------#

    def compute_advantages(self, obs, actions):

        print("diffusion_idql.py: IDQLDiffusion.compute_advantages()")

        # get current Q-function, stop gradient
        with torch_no_grad() as tape:
            current_q1, current_q2 = self.target_q(obs, actions)
        q = torch_min(current_q1, current_q2)

        # get the current V-function
        v = self.critic_v(obs)
        
        v = torch_reshape(v, [-1])

        # compute advantage
        adv = q - v
        return adv

    def loss_critic_v(self, obs, actions):

        print("diffusion_idql.py: IDQLDiffusion.loss_critic_v()")

        adv = self.compute_advantages(obs, actions)

        # get the value loss
        v_loss = torch_mean( expectile_loss(adv) )
        return v_loss

    def loss_critic_q(self, obs, next_obs, actions, rewards, terminated, gamma):

        print("diffusion_idql.py: IDQLDiffusion.loss_critic_q()")

        # get current Q-function
        current_q1, current_q2 = self.critic_q(obs, actions)

        # get the next V-function, stop gradient
        with torch_no_grad() as tape:
            next_v = self.critic_v(next_obs)

        # terminal state mask
        mask = 1 - terminated

        # flatten
        rewards = torch_tensor_view(rewards, [-1])
        next_v = torch_tensor_view(next_v, [-1])
        mask = torch_tensor_view(mask, [-1])

        # target value
        discounted_q = rewards + gamma * next_v * mask

        # Update critic
        q_loss = torch_mean( (current_q1 - discounted_q) ** 2 ) + torch_mean( (current_q2 - discounted_q) ** 2 )
        return q_loss


    def update_target_critic(self, tau):

        print("diffusion_idql.py: IDQLDiffusion.update_target_critic()")

        for target_param, source_param in zip(self.target_q.trainable_variables, self.critic_q.trainable_variables):
            target_param.assign(target_param * (1.0 - tau) + source_param * tau)


    def p_losses(
        self,
        x_start,
        cond,
        t, training = True
    ):
        """not reward-weighted, same as diffusion.py"""

        print("diffusion_idql.py: IDQLDiffusion.p_losses()")

        # Forward process
        noise = torch_randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict

        x_recon = self.network( [x_noisy, t, cond['state']], training=training)

        # Loss with mask           
        if self.predict_epsilon:
            loss = torch_mse_loss(x_recon, noise)
        else:
            loss = torch_mse_loss(x_recon, x_start)

        return loss

    # ---------- Sampling ----------#``



    @tf.function
    def call(
        self,
        cond,
        deterministic=False,
        num_sample=10,
        critic_hyperparam=0.7,  # sampling weight for implicit policy
        use_expectile_exploration=True,
    ):
        """assume state-only, no rgb in cond"""

        with torch_no_grad() as tape:

            print("diffusion_idql.py: IDQLDiffusion.forward()")

            # repeat obs num_sample times along dim 0
            cond_shape_repeat_dims = tuple(1 for _ in cond["state"].shape)
            B, T, D = cond["state"].shape
            S = num_sample

            cond_repeat = torch_tensor_repeat( cond["state"][None], num_sample, *cond_shape_repeat_dims)

            cond_repeat = torch_tensor_view(cond_repeat, -1, T, D)


            # for eval, use less noisy samples --- there is still DDPM noise, but final action uses small min_sampling_std
            samples = super(IDQLDiffusion, self).call(
                {"state": cond_repeat},
                deterministic=deterministic,
            )

            _, H, A = samples.shape

            # get current Q-function
            current_q1, current_q2 = self.target_q({"state": cond_repeat}, samples)

            q = torch_min(current_q1, current_q2)
            q = torch_tensor_view(q, [S, B])


            # Use argmax
            if deterministic or (not use_expectile_exploration):
                # gather the best sample -- filter out suboptimal Q during inference
                best_indices = torch_argmax(q, 0)
                samples_expanded = torch_tensor_view(samples, [S, B, H, A])

                # dummy dimension @ dim 0 for batched indexing
                sample_indices = best_indices[None, :, None, None]  # [1, B, 1, 1]
                sample_indices = torch_tensor_repeat(sample_indices, S, 1, H, A)

                samples_best = torch_gather(samples_expanded, 0, sample_indices)

            # Sample as an implicit policy for exploration
            else:
                # get the current value function for probabilistic exploration
                current_v = self.critic_v({"state": cond_repeat})

                v = torch_tensor_view(current_v, [S, B])

                adv = q - v

                # Compute weights for sampling
                samples_expanded = torch_tensor_view(samples, [S, B, H, A])

                # expectile exploration policy
                tau_weights = torch_where(adv > 0, critic_hyperparam, 1 - critic_hyperparam)
                tau_weights = tau_weights / torch_sum(tau_weights, 0)  # normalize

                assert len( tau_weights.shape.as_list() ) >= 2, "tau_weights.shape.as_list() should be at least 2"

                # select a sample from DP probabilistically -- sample index per batch and compile
                
                sample_indices = torch_multinomial(tf.transpose(tau_weights), 1)  # [B, 1]
                
                # dummy dimension @ dim 0 for batched indexing
                sample_indices = sample_indices[None, :, None]  # [1, B, 1, 1]
                sample_indices = torch_tensor_repeat(sample_indices, S, 1, H, A)

                samples_best = torch_gather(samples_expanded, 0, sample_indices)


            # squeeze dummy dimension
            samples = samples_best[0]
            return samples


































