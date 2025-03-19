"""
Actor and Critic models for model-free online RL with DIffusion POlicy (DIPO).

"""

import tensorflow as tf

import logging
import copy

log = logging.getLogger(__name__)

from model.diffusion.diffusion import DiffusionModel
from model.diffusion.sampling import make_timesteps

from util.torch_to_tf import torch_no_grad, torch_min, torch_tensor_view, torch_mean,\
torch_randn, torch_exp, torch_clip, torch_zeros_like, torch_clamp, torch_randn_like


class DIPODiffusion(DiffusionModel):

    def __init__(
        self,
        actor,
        critic,
        use_ddim=False,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        **kwargs,
    ):

        print("diffusion_dipo.py: DIPODiffusion.__init__()")

        super().__init__(network=actor, use_ddim=use_ddim, **kwargs)
        assert not self.use_ddim, "DQL does not support DDIM"
        self.critic = critic


        # target critic
        self.critic_target = copy.deepcopy(self.critic)

        # reassign actor
        self.actor = self.network

        # target actor
        self.actor_target = copy.deepcopy(self.actor)

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

    # ---------- RL training ----------#

    def loss_critic(self, obs, next_obs, actions, rewards, terminated, gamma):

        print("diffusion_dipo.py: DIPODiffusion.loss_critic()")

        # get current Q-function
        current_q1, current_q2 = self.critic(obs, actions)

        # Get next Q-function
        with torch_no_grad() as tape:
            next_actions = self.call(
                cond=next_obs,
                deterministic=False,
            )  # forward() has no gradient, which is desired here.
            next_q1, next_q2 = self.critic_target(next_obs, next_actions)
            next_q = torch_min(next_q1, next_q2)

            # terminal state mask
            mask = 1 - terminated

            # flatten
            rewards = torch_tensor_view(rewards, [-1])
            next_q = torch_tensor_view(next_q, [-1])
            mask = torch_tensor_view(mask, [-1])

            # target value
            target_q = rewards + gamma * next_q * mask

        # Update critic loss
        loss_critic = torch_mean( (current_q1 - target_q) ** 2 ) + torch_mean(
            (current_q2 - target_q) ** 2
        )
        return loss_critic



    def update_target_critic(self, tau):
        print("diffusion_dipo.py: DIPODiffusion.update_target_critic()")

        for target_param, source_param in zip(
            self.critic_target.trainable_variables, self.critic.trainable_variables
        ):
            target_param.assign(target_param * (1.0 - tau) + source_param * tau)

    def update_target_actor(self, tau):
        print("diffusion_dipo.py: DIPODiffusion.update_target_actor()")

        for target_param, source_param in zip(
            self.actor_target.trainable_variables, self.actor.trainable_variables
        ):
            target_param.assign(target_param * (1.0 - tau) + source_param * tau)


    # ---------- Sampling ----------#``

    # override
    @tf.function
    def call(
        self,
        cond,
        deterministic=False,
    ):
        """Use target actor"""

        with torch_no_grad() as tape:

            print("diffusion_dipo.py: DIPODiffusion.call()")

            B = tf.shape(cond["state"])[0]

            # Loop
            x = torch_randn((B, self.horizon_steps, self.action_dim))

            t_all = list(reversed(range(self.denoising_steps)))
            for i, t in enumerate(t_all):
                t_b = make_timesteps(B, t)
                mean, logvar = self.p_mean_var(
                    x=x,
                    t=t_b,
                    cond_state=cond['state'],
                    network_override=self.actor_target,
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
                if self.final_action_clip_value is not None and i == len(t_all) - 1:
                    x = torch_clamp(x, -self.final_action_clip_value, self.final_action_clip_value)

            return x











