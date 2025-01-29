"""
Reward-weighted regression (RWR) for diffusion policy.

"""


import tensorflow as tf

import logging
import einops

log = logging.getLogger(__name__)


from model.diffusion.diffusion import DiffusionModel
from model.diffusion.sampling import make_timesteps


from util.torch_to_tf import torch_randn_like, torch_mse_loss, torch_mean, torch_randn

from util.torch_to_tf import torch_exp, torch_zeros_like, torch_clip, torch_clamp


class RWRDiffusion(DiffusionModel):

    def __init__(
        self,
        use_ddim=False,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        **kwargs,
    ):

        print("diffusion_rwr.py: RWRDiffusion.__init__()")

        super().__init__(use_ddim=use_ddim, **kwargs)
        assert not self.use_ddim, "RWR does not support DDIM"

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

    # ---------- RL training ----------#

    # override
    def p_losses(
        self,
        x_start,
        cond,
        rewards,
        t,
    ):
        """reward-weighted"""

        print("diffusion_rwr.py: RWRDiffusion.p_losses()")

        # device = x_start.device
        # Forward process
        # noise = tf.random.normal(shape=tf.shape(x_start), dtype=x_start.dtype)
        noise = torch_randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond)

        if self.predict_epsilon:
            loss = torch_mse_loss(x_recon, noise, 'none')
        else:
            loss = torch_mse_loss(x_recon, x_start, 'none')


        loss = einops.reduce(loss, "b h d -> b", "mean")
        loss *= rewards
        return torch_mean(loss)
    


    # ---------- Sampling ----------#

    @tf.function
    def call(self, cond, deterministic=False):
    # # override
    # @torch.no_grad()
    # def forward(
    #     self,
    #     cond,
    #     deterministic=False,
    # ):
        """Modifying denoising schedule"""

        print("diffusion_rwr.py: RWRDiffusion.forward()")

        # device = self.betas.device
        # B = len(cond["state"])

        B = cond["state"].shape[0]

        # Initialize x
        # x = tf.random.normal((B, self.horizon_steps, self.action_dim), dtype=tf.float32)
        x = torch_randn( (B, self.horizon_steps, self.action_dim) )

        t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
            )

            std = torch_exp(0.5 * logvar)


            # Determine noise level
            if deterministic and t == 0:
                std = torch_zeros_like(std)
            elif deterministic:
                std = torch_clip(std, 1e-3, float('inf'))
            else:
                std = torch_clip(std, self.min_sampling_denoising_std, float('inf'))

            # Add noise
            noise = torch_randn_like(x)
            noise = torch_clamp(noise, -self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            # Clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch_clamp(x, -self.final_action_clip_value, self.final_action_clip_value)
        
        return x


