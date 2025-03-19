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


from util.torch_to_tf import torch_tensor_detach, \
    torch_mean, torch_tensor_view, torch_min, torch_abs, \
        torch_square, torch_randn, torch_exp, torch_clip,\
        torch_zeros_like, torch_no_grad, torch_randn_like,\
        torch_clamp




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
        
        # # target critic
        self.critic_target = copy.deepcopy(self.critic)



        # reassign actor
        self.actor = self.network
        self.build_actor(self.actor)

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

    # ---------- RL training ----------#

    def loss_critic(self, obs, next_obs, actions, rewards, terminated, gamma):

        print("diffusion_dql.py: DQLDiffusion.loss_critic()")

        # get current Q-function
        current_q1, current_q2 = self.critic(obs, actions)

        with torch_no_grad() as tape:
            # get next Q-function
            next_actions = self.call(
                cond=next_obs,
                deterministic=False,
            )  # forward() has no gradient, which is desired here.
            next_q1, next_q2 = self.critic_target(next_obs, next_actions)

            next_q = torch_min(next_q1, other=next_q2)

            # terminal state mask
            mask = 1 - terminated

            # flatten
            rewards = torch_tensor_view(rewards, -1)
            next_q = torch_tensor_view(next_q, -1)
            mask = torch_tensor_view(mask, -1)

            # target value
            target_q = rewards + gamma * next_q * mask

        # Update critic
        loss_critic = torch_mean( (current_q1 - target_q) ** 2 ) + torch_mean(
            (current_q2 - target_q) ** 2
        )
        return loss_critic

    def loss_actor(self, obs, eta, act_steps, training=True):

        print("diffusion_dql.py: DQLDiffusion.loss_actor()")

        action_new = self.forward_train(
            cond=obs,
            deterministic=False,
        )[
            :, :act_steps
        ]  # with gradient
        q1, q2 = self.critic(obs, action_new)
        bc_loss = self.loss_ori(training, action_new, obs)
        if np.random.uniform() > 0.5:
            q_loss = -torch_mean(q1) / torch_tensor_detach( torch_mean(torch_abs(q2)) )

        else:
            q_loss = -torch_mean(q2) / torch_tensor_detach( torch_mean(torch_abs(q1)) )
        actor_loss = bc_loss + eta * q_loss
        return actor_loss

    def update_target_critic(self, tau):

        print("diffusion_dql.py: DQLDiffusion.update_target_critic()")

        # for target_param, source_param in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
        for target_param, source_param in zip(self.critic_target.variables, self.critic.variables):
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
        with torch_no_grad() as tape:

            print("diffusion_dql.py: DQLDiffusion.call()")

            B = tf.shape(cond["state"])[0]

            x = torch_randn( (B, self.horizon_steps, self.action_dim) )

            t_all = list(reversed(range(self.denoising_steps)))
            for i, t in enumerate(t_all):
                t_b = make_timesteps(B, t)
                mean, logvar = self.p_mean_var(
                    x=x,
                    t=t_b,
                    cond_state=cond['state'],
                )
                std = torch_exp(0.5 * logvar)

                # Determine the noise level
                if deterministic and t == 0:
                    std = torch_zeros_like(std)
                elif deterministic:
                    std = torch_clip(std, min = 1e-3)
                else:
                    std = torch_clip(std, min = self.min_sampling_denoising_std)
                
                noise = torch_randn_like(x)
                noise = torch_clamp(noise, -self.randn_clip_value, self.randn_clip_value)
                
                x = mean + std * noise

                # clamp action at final step
                if self.final_action_clip_value is not None and i == len(t_all) - 1:
                    x = torch_clamp(x, -self.final_action_clip_value, self.final_action_clip_value)

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

        B = tf.shape(cond["state"])[0]

        # Loop
        x = torch_randn([B, self.horizon_steps, self.action_dim] )

        t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond_state=cond['state'],
            )
            std = torch_exp(0.5 * logvar)

            # Determine the noise level
            if deterministic and t == 0:
                std = torch_zeros_like(std)
            elif deterministic:  # For DDPM, sample with noise
                std = torch_clip(std, 1e-3, float('inf'))
            else:
                std = torch_clip(std, self.min_sampling_denoising_std, float('inf'))
            
            noise = torch_randn_like(x)
            noise = torch_clamp(noise, -self.randn_clip_value, self.randn_clip_value)            
            
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value and i == len(t_all) - 1:
                x = torch_clamp(x, -1, 1)
        return x
