"""
Diffusion Q-Learning (DQL)

"""

import tensorflow as tf

import logging
import numpy as np
import copy

log = logging.getLogger(__name__)

from model.diffusion.diffusion import DiffusionModel
from model.diffusion.sampling import make_timesteps


class DQLDiffusion(DiffusionModel):

    def __init__(
        self,
        actor,
        critic,
        use_ddim=False,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        **kwargs,
    ):

        print("diffusion_dql.py: DQLDiffusion.__init__()")

        super().__init__(network=actor, use_ddim=use_ddim, **kwargs)
        assert not self.use_ddim, "DQL does not support DDIM"
        self.critic = critic
        
        # target critic
        self.critic_target = copy.deepcopy(self.critic)

        # reassign actor
        self.actor = self.network

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

    # ---------- RL training ----------#

    def loss_critic(self, obs, next_obs, actions, rewards, terminated, gamma):

        print("diffusion_dql.py: DQLDiffusion.loss_critic()")

        # get current Q-function
        current_q1, current_q2 = self.critic(obs, actions)

        # get next Q-function
        next_actions = self.call(
            cond=next_obs,
            deterministic=False,
        )  # forward() has no gradient, which is desired here.
        next_q1, next_q2 = self.critic_target(next_obs, next_actions)
        next_q = tf.minimum(next_q1, next_q2)

        # terminal state mask
        mask = 1 - terminated

        # flatten
        rewards = tf.reshape(rewards, [-1])
        next_q = tf.reshape(next_q, [-1])
        mask = tf.reshape(mask, [-1])

        # target value
        target_q = rewards + gamma * next_q * mask

        # Update critic
        loss_critic = tf.reduce_mean(tf.square(current_q1 - target_q)) + tf.reduce_mean(
            tf.square(current_q2 - target_q)
        )
        return loss_critic

    def loss_actor(self, obs, eta, act_steps):

        print("diffusion_dql.py: DQLDiffusion.loss_actor()")

        action_new = self.forward_train(
            cond=obs,
            deterministic=False,
        )[
            :, :act_steps
        ]  # with gradient
        q1, q2 = self.critic(obs, action_new)
        bc_loss = self.loss(action_new, obs)
        if np.random.uniform() > 0.5:
            q_loss = -tf.reduce_mean(q1) / tf.reduce_mean(tf.abs(q2))

        else:
            q_loss = -tf.reduce_mean(q2) / tf.reduce_mean(tf.abs(q1))
        actor_loss = bc_loss + eta * q_loss
        return actor_loss

    def update_target_critic(self, tau):

        print("diffusion_dql.py: DQLDiffusion.update_target_critic()")

        for target_param, source_param in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
            target_param.assign(
                target_param * (1.0 - tau) + source_param * tau
            )


    # ---------- Sampling ----------#``

    # override
    @tf.function
    def call(
        self,
        cond,
        deterministic=False,
    ):

        print("diffusion_dql.py: DQLDiffusion.call()")

        # device = self.betas.device
        # B = len(cond["state"])
        B = tf.shape(cond["state"])[0]

        # Loop
        x = tf.random.normal([B, self.horizon_steps, self.action_dim], dtype=tf.float32)
        t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
            )
            std = tf.exp(0.5 * logvar)

            # Determine the noise level
            if deterministic and t == 0:
                std = tf.zeros_like(std)
            elif deterministic:
                std = tf.clip_by_value(std, 1e-3, float('inf'))
            else:
                std = tf.clip_by_value(std, self.min_sampling_denoising_std, float('inf'))
            
            noise = tf.random.normal(tf.shape(x), dtype=tf.float32)
            noise = tf.clip_by_value(noise, -self.randn_clip_value, self.randn_clip_value)
            
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = tf.clip_by_value(x, -self.final_action_clip_value, self.final_action_clip_value)

        return x

    def forward_train(
        self,
        cond,
        deterministic=False,
    ):
        """
        Differentiable forward pass used in actor training.
        """

        print("diffusion_dql.py: DQLDiffusion.forward_train()")

        # device = self.betas.device
        B = tf.shape(cond["state"])[0]

        # Loop
        # x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        x = tf.random.normal([B, self.horizon_steps, self.action_dim], dtype=tf.float32)

        t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
            )
            std = tf.exp(0.5 * logvar)

            # Determine the noise level
            if deterministic and t == 0:
                std = tf.zeros_like(std)
            elif deterministic:  # For DDPM, sample with noise
                std = tf.clip_by_value(std, 1e-3, float('inf'))
            else:
                std = tf.clip_by_value(std, self.min_sampling_denoising_std, float('inf'))
            
            noise = tf.random.normal(tf.shape(x), dtype=tf.float32)
            noise = tf.clip_by_value(noise, -self.randn_clip_value, self.randn_clip_value)            
            
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value and i == len(t_all) - 1:
                x = tf.clip_by_value(x, -1, 1)
        return x
