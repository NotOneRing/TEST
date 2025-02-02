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

from util.torch_to_tf import torch_no_grad, torch_randn_like, torch_randint,\
torch_tensor_long, torch_sum, torch_mse_loss, torch_mean, torch_stack, torch_min,\
torch_tensor_view, torch_tensor_detach



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

        print("type(critic) = ", critic)
        print("critic = ", critic)
        # target critic
        self.target_q = copy.deepcopy(critic)

        # assign actor
        self.actor = self.network

    # ---------- RL training ----------#

    def loss_actor(self, obs, actions, q_grad_coeff):

        print("diffusion_qsm.py: QSMDiffusion.loss_actor()")

        x_start = actions

        B = tf.shape(x_start)[0]

        # Forward process
        noise = torch_randn_like(x_start)
        t = torch_tensor_long( torch_randint( low=0, high=self.denoising_steps, size=(B,) ) )  # sample random denoising time index

        # get current value for noisy actions as the code does --- the algorthm block in the paper is wrong, it says using a_t, the final denoised action
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Compute Q values for noisy actions
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_noisy)
            # tape.watch(x_noisy)
            current_q1, current_q2 = self.critic_q(obs, x_noisy, training=True)
            q1_sum = torch_sum(current_q1)
            q2_sum = torch_sum(current_q2)


        # Compute gradients dQ/da
        gradient_q1 = tape.gradient(q1_sum, x_noisy)
        gradient_q2 = tape.gradient(q2_sum, x_noisy)
        gradient_q = torch_tensor_detach( torch_mean(torch_stack((gradient_q1, gradient_q2), dim=0), dim=0) )
        # # Compute dQ/da|a=noise_actions
        # gradient_q1 = torch.autograd.grad(current_q1.sum(), x_noisy)[0]
        # gradient_q2 = torch.autograd.grad(current_q2.sum(), x_noisy)[0]
        # gradient_q = torch.stack((gradient_q1, gradient_q2), 0).mean(0).detach()

        # Predict noise from noisy actions
        # x_recon = self.network([x_noisy, t, obs], training=True)
        x_recon = self.network([x_noisy, t, obs['state']], training=True)

        # Loss with mask - align predicted noise with critic gradient of noisy actions
        # Note: the gradient of mu wrt. epsilon has a negative sign
        loss = torch_mse_loss(-x_recon, q_grad_coeff * gradient_q)
        return loss

    def loss_critic(self, obs, next_obs, actions, rewards, terminated, gamma):

        print("diffusion_qsm.py: QSMDiffusion.loss_critic()")

        # get current Q-function
        # current_q1, current_q2 = self.critic_q([obs, actions], training=True)
        current_q1, current_q2 = self.critic_q(obs, actions, training=True)

        # get next Q-function - with noise, same as QSM https://github.com/Alescontrela/score_matching_rl/blob/f02a21969b17e322eb229ceb2b0f5a9111b1b968/jaxrl5/agents/score_matching/score_matching_learner.py#L193
        next_actions = self.call(cond=next_obs, deterministic=False)

        with torch_no_grad() as tape:
            next_q1, next_q2 = self.target_q(next_obs, next_actions, training=False)

        next_q = torch_min(next_q1, other=next_q2)

        # terminal state mask
        mask = 1 - terminated

        # flatten
        rewards = torch_tensor_view(rewards, [-1] )
        next_q = torch_tensor_view(next_q, [-1])
        mask = torch_tensor_view(mask, [-1])

        # target value
        discounted_q = rewards + gamma * next_q * mask

        # Update critic
        loss_critic = torch_mean( (current_q1 - discounted_q) ** 2 ) + torch_mean(
            (current_q2 - discounted_q) ** 2
        )

        return loss_critic

    def update_target_critic(self, tau):

        print("diffusion_qsm.py: QSMDiffusion.update_target_critic()")

        for target_var, source_var in zip(self.target_q.trainable_variables, self.critic_q.trainable_variables):
            target_var.assign(tau * source_var + (1.0 - tau) * target_var)
