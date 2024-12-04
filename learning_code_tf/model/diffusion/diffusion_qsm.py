"""
QSM (Q-Score Matching) for diffusion policy.

"""

import logging
import copy

# import torch
# import torch.nn.functional as F

import tensorflow as tf

log = logging.getLogger(__name__)

from model.diffusion.diffusion_rwr import RWRDiffusion


class QSMDiffusion(RWRDiffusion):

    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):

        print("diffusion_qsm.py: QSMDiffusion.__init__()")

        super().__init__(network=actor, **kwargs)
        self.critic_q = critic


        # target critic
        self.target_q = copy.deepcopy(critic)

        # assign actor
        self.actor = self.network

    # ---------- RL training ----------#

    def loss_actor(self, obs, actions, q_grad_coeff):

        print("diffusion_qsm.py: QSMDiffusion.loss_actor()")

        x_start = actions

        batch_size = tf.shape(x_start)[0]

        # Forward process
        noise = tf.random.normal(tf.shape(x_start))
        t = tf.random.uniform(
            [batch_size], minval=0, maxval=self.denoising_steps, dtype=tf.int32
        )  # Random time step

        # get current value for noisy actions as the code does --- the algorthm block in the paper is wrong, it says using a_t, the final denoised action
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Compute Q values for noisy actions
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            tape1.watch(x_noisy)
            tape2.watch(x_noisy)
            current_q1, current_q2 = self.critic_q([obs, x_noisy], training=True)

        # Compute gradients dQ/da
        gradient_q1 = tape1.gradient(current_q1, x_noisy)
        gradient_q2 = tape2.gradient(current_q2, x_noisy)
        gradient_q = tf.reduce_mean(tf.stack([gradient_q1, gradient_q2]), axis=0)
        # # Compute dQ/da|a=noise_actions
        # gradient_q1 = torch.autograd.grad(current_q1.sum(), x_noisy)[0]
        # gradient_q2 = torch.autograd.grad(current_q2.sum(), x_noisy)[0]
        # gradient_q = torch.stack((gradient_q1, gradient_q2), 0).mean(0).detach()

        # Predict noise from noisy actions
        x_recon = self.actor([x_noisy, t, obs], training=True)

        # Loss with mask - align predicted noise with critic gradient of noisy actions
        # Note: the gradient of mu wrt. epsilon has a negative sign
        loss = tf.reduce_mean(tf.square(-x_recon - q_grad_coeff * gradient_q))
        return loss

    def loss_critic(self, obs, next_obs, actions, rewards, terminated, gamma):

        print("diffusion_qsm.py: QSMDiffusion.loss_critic()")

        # get current Q-function
        current_q1, current_q2 = self.critic_q([obs, actions], training=True)

        # get next Q-function - with noise, same as QSM https://github.com/Alescontrela/score_matching_rl/blob/f02a21969b17e322eb229ceb2b0f5a9111b1b968/jaxrl5/agents/score_matching/score_matching_learner.py#L193
        next_actions = self.call(cond=next_obs, deterministic=False)
        next_q1, next_q2 = self.target_q([next_obs, next_actions], training=False)
        next_q = tf.minimum(next_q1, next_q2)

        # terminal state mask
        mask = 1 - terminated

        # flatten
        rewards = tf.reshape(rewards, [-1] )
        next_q = tf.reshape(next_q, [-1])
        mask = tf.reshape(mask, [-1])

        # target value
        discounted_q = rewards + gamma * next_q * mask

        # Update critic
        loss_critic = tf.reduce_mean(tf.square(current_q1 - discounted_q)) + tf.reduce_mean(
            tf.square(current_q2 - discounted_q)
        )

        return loss_critic

    def update_target_critic(self, tau):

        print("diffusion_qsm.py: QSMDiffusion.update_target_critic()")

        for target_var, source_var in zip(self.target_q.trainable_variables, self.critic_q.trainable_variables):
            target_var.assign(tau * source_var + (1.0 - tau) * target_var)
