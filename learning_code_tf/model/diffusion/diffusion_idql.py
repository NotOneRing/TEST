"""
Implicit diffusion Q-learning (IDQL) for diffusion policy.

"""

import logging

import einops
import copy

import tensorflow as tf

log = logging.getLogger(__name__)

from model.diffusion.diffusion_rwr import RWRDiffusion


def expectile_loss(diff, expectile=0.8):

    print("diffusion_idql.py: expectile_loss()")

    # weight = torch.where(diff > 0, expectile, (1 - expectile))
    weight = tf.where(diff > 0, expectile, (1 - expectile))

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
        self.critic_q = critic_q.to(self.device)
        self.target_q = copy.deepcopy(critic_q)
        self.critic_v = critic_v.to(self.device)

        # assign actor
        self.actor = self.network

    # ---------- RL training ----------#

    def compute_advantages(self, obs, actions):

        print("diffusion_idql.py: IDQLDiffusion.compute_advantages()")

        # get current Q-function, stop gradient

        current_q1, current_q2 = self.target_q(obs, actions)
        q = tf.minimum(current_q1, current_q2)

        # get the current V-function
        v = self.critic_v(obs)
        
        v = tf.reshape(v, [-1])

        # compute advantage
        adv = q - v
        return adv

    def loss_critic_v(self, obs, actions):

        print("diffusion_idql.py: IDQLDiffusion.loss_critic_v()")

        adv = self.compute_advantages(obs, actions)

        # get the value loss
        v_loss = expectile_loss(adv).mean()
        return v_loss

    def loss_critic_q(self, obs, next_obs, actions, rewards, terminated, gamma):

        print("diffusion_idql.py: IDQLDiffusion.loss_critic_q()")

        # get current Q-function
        current_q1, current_q2 = self.critic_q(obs, actions)

        # get the next V-function, stop gradient
        next_v = self.critic_v(next_obs)

        # terminal state mask
        mask = 1 - terminated

        # flatten
        rewards = tf.reshape(rewards, [-1])
        next_v = tf.reshape(next_v, [-1])
        mask = tf.reshape(mask, [-1])

        # target value
        discounted_q = rewards + gamma * next_v * mask

        # Update critic
        q_loss = tf.reduce_mean(tf.square(current_q1 - discounted_q)) + tf.reduce_mean(tf.square(current_q2 - discounted_q))
        return q_loss


    def update_target_critic(self, tau):

        print("diffusion_idql.py: IDQLDiffusion.update_target_critic()")

        for target_param, source_param in zip(self.target_q.trainable_variables, self.critic_q.trainable_variables):
            target_param.assign(target_param * (1.0 - tau) + source_param * tau)


    # override
    def p_losses(
        self,
        x_start,
        cond,
        t,
    ):
        """not reward-weighted, same as diffusion.py"""

        print("diffusion_idql.py: IDQLDiffusion.p_losses()")

        # Forward process
        noise = tf.random.normal(shape=tf.shape(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond)

        # Loss with mask           
        if self.predict_epsilon:
            loss = tf.reduce_mean(tf.square(x_recon - noise))
        else:
            loss = tf.reduce_mean(tf.square(x_recon - x_start))

        return loss

    # ---------- Sampling ----------#``




    def call(
        self,
        cond,
        deterministic=False,
        num_sample=10,
        critic_hyperparam=0.7,  # sampling weight for implicit policy
        use_expectile_exploration=True,
    ):
        """assume state-only, no rgb in cond"""

        print("diffusion_idql.py: IDQLDiffusion.forward()")

        # repeat obs num_sample times along dim 0
        cond_shape_repeat_dims = tuple(1 for _ in cond["state"].shape)
        B, T, D = cond["state"].shape
        S = num_sample

        cond_repeat = cond["state"][None].repeat(num_sample, *cond_shape_repeat_dims)

        cond_repeat = tf.tile(cond["state"][None], [num_sample] + [1] * len(cond["state"].shape))

        cond_repeat = tf.reshape(cond_repeat, [-1, T, D])  # [B*S, T, D]

        # for eval, use less noisy samples --- there is still DDPM noise, but final action uses small min_sampling_std
        samples = super(IDQLDiffusion, self).call(
            {"state": cond_repeat},
            deterministic=deterministic,
        )

        _, H, A = samples.shape

        # get current Q-function
        current_q1, current_q2 = self.target_q({"state": cond_repeat}, samples)

        q = tf.minimum(current_q1, current_q2)
        q = tf.reshape(q, [S, B])


        # Use argmax
        if deterministic or (not use_expectile_exploration):
            # gather the best sample -- filter out suboptimal Q during inference
            best_indices = tf.argmax(q, axis=0)
            samples_expanded = tf.reshape(samples, [S, B, H, A])

            # dummy dimension @ dim 0 for batched indexing
            sample_indices = best_indices[None, :, None, None]  # [1, B, 1, 1]
            sample_indices = tf.tile(sample_indices, [S, 1, H, A])

            # samples_best = torch.gather(samples_expanded, 0, sample_indices)

            # samples_best = tf.gather(samples_expanded, sample_indices, axis=0)

            samples_best = tf.gather(samples_expanded, sample_indices, axis=0, batch_dims=1)

        # Sample as an implicit policy for exploration
        else:
            # get the current value function for probabilistic exploration
            current_v = self.critic_v({"state": cond_repeat})

            v = tf.reshape(current_v, [S, B])

            adv = q - v

            # select a sample from DP probabilistically -- sample index per batch and compile
            sample_indices = torch.multinomial(tau_weights.T, 1)  # [B, 1]

            # dummy dimension @ dim 0 for batched indexing
            sample_indices = sample_indices[None, :, None]  # [1, B, 1, 1]
            sample_indices = sample_indices.repeat(S, 1, H, A)

            samples_best = torch.gather(samples_expanded, 0, sample_indices)

            # Compute weights for sampling
            samples_expanded = tf.reshape(samples, [S, B, H, A])

            # expectile exploration policy
            tau_weights = tf.where(adv > 0, critic_hyperparam, 1 - critic_hyperparam)
            tau_weights = tau_weights / tf.reduce_sum(tau_weights, axis=0)  # normalize

            # select a sample from DP probabilistically -- sample index per batch and compile
            sample_indices = tf.random.categorical(tau_weights.T, 1)  # [B, 1]

            # dummy dimension @ dim 0 for batched indexing
            sample_indices = tf.expand_dims(sample_indices, 0)  # [1, B, 1, 1]
            sample_indices = tf.tile(sample_indices, [S, 1, H, A])

            samples_best = tf.gather(samples_expanded, sample_indices, axis=0)


        # squeeze dummy dimension
        samples = samples_best[0]
        return samples


































