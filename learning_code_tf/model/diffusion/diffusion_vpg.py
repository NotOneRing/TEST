"""
Policy gradient with diffusion policy. VPG: vanilla policy gradient

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""

import copy
import logging

log = logging.getLogger(__name__)

from model.diffusion.diffusion import DiffusionModel, Sample
from model.diffusion.sampling import make_timesteps, extract

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

from util.torch_to_tf import torch_flatten, torch_arange, Normal

from util.torch_to_tf import torch_no_grad, torch_where


class VPGDiffusion(DiffusionModel):

    def __init__(
        self,
        actor,
        critic,
        ft_denoising_steps,
        ft_denoising_steps_d=0,
        ft_denoising_steps_t=0,
        network_path=None,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        # eta in DDIM
        eta=None,
        learn_eta=False,
        **kwargs,
    ):

        print("diffusion_vpg.py: VPGDiffusion.__init__()")

        super().__init__(
            network=actor,
            network_path=network_path,
            **kwargs,
        )
        assert ft_denoising_steps <= self.denoising_steps
        assert ft_denoising_steps <= self.ddim_steps if self.use_ddim else True
        assert not (learn_eta and not self.use_ddim), "Cannot learn eta with DDPM."

        self.actor = actor
        self.critic = critic

        # Number of denoising steps to use with fine-tuned model. Thus denoising_step - ft_denoising_steps is the number of denoising steps to use with original model.
        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = ft_denoising_steps_d  # annealing step size
        self.ft_denoising_steps_t = ft_denoising_steps_t  # annealing interval
        self.ft_denoising_steps_cnt = 0

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

        # Minimum std used in calculating denoising logprobs - for stability
        self.min_logprob_denoising_std = min_logprob_denoising_std

        # Learnable eta
        self.learn_eta = learn_eta
        if eta is not None:
            self.eta = eta
            if not learn_eta:
                for param in self.eta.trainable_variables:
                    param._trainable = False

                logging.info("Turned off gradients for eta")


        # Re-name network to actor
        self.actor = self.network

        # Make a copy of the original model
        self.actor_ft = copy.deepcopy(self.actor)
        logging.info("Cloned model for fine-tuning")

        # Turn off gradients for original model
        for var in self.actor.trainable_variables:
            var._trainable = False
            
        logging.info("Turned off gradients of the pretrained network")
        
        logging.info(
            f"Number of finetuned parameters: {sum([tf.size(v) for v in self.actor_ft.trainable_variables])}"
        )

        # Value function
        self.critic = critic

        if network_path is not None:
            # # checkpoint = torch.load(
            # #     network_path, map_location=self.device, weights_only=True
            # # )
            # checkpoint = tf.train.Checkpoint(model=self.network)

            # latest_checkpoint = tf.train.latest_checkpoint(network_path)

            # if "ema" not in checkpoint:  # load trained RL model
            #     # self.load_state_dict(checkpoint["model"], strict=False)

            #     checkpoint.restore(latest_checkpoint)

            #     logging.info("Loaded critic from %s", network_path)

            print("self.network_path is not None")

            loadpath = network_path

            print("loadpath = ", loadpath)

            # self.model.load_weights(loadpath)
            # self.ema_model.load_weights(loadpath.replace(".h5", "_ema.h5"))



            from model.diffusion.mlp_diffusion import DiffusionMLP
            from model.common.mlp import MLP, ResidualMLP
            from model.diffusion.modules import SinusoidalPosEmb
            from model.common.modules import SpatialEmb, RandomShiftsAug
            from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish

            from tensorflow.keras.utils import get_custom_objects

            # Register your custom class with Keras
            get_custom_objects().update({
                'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
                'DiffusionMLP': DiffusionMLP,
                'VPGDiffusion': VPGDiffusion,
                'SinusoidalPosEmb': SinusoidalPosEmb,  # 假设 SinusoidalPosEmb 是你自定义的层
                'MLP': MLP,                            # 自定义的 MLP 层
                'ResidualMLP': ResidualMLP,            # 自定义的 ResidualMLP 层
                'nn_Sequential': nn_Sequential,        # 自定义的 Sequential 类
                'nn_Linear': nn_Linear,
                'nn_LayerNorm': nn_LayerNorm,
                'nn_Dropout': nn_Dropout,
                'nn_ReLU': nn_ReLU,
                'nn_Mish': nn_Mish,
                'SpatialEmb': SpatialEmb,
                'RandomShiftsAug': RandomShiftsAug,
             })


            self.model = tf.keras.models.load_model(loadpath, custom_objects=get_custom_objects())
            # self.ema_model = tf.keras.models.load_model(loadpath.replace(".h5", "_ema.h5"), custom_objects=get_custom_objects())
            self.ema_model = tf.keras.models.load_model(loadpath.replace(".keras", "_ema.keras"), custom_objects=get_custom_objects())




    def get_config(self):
        """
        Returns the configuration of the VPGDiffusion instance as a dictionary.
        
        Returns:
            dict: Configuration dictionary for the VPGDiffusion instance.
        """
        
        config = super().get_config()  # Get the config from the parent (DiffusionModel)
        
        # Add the configuration for the VPGDiffusion-specific attributes
        config.update({
            'actor': self.actor,  # Actor model (could be any object, make sure it's serializable)
            'critic': self.critic,  # Critic model (could be any object, make sure it's serializable)
            'ft_denoising_steps': self.ft_denoising_steps,
            'ft_denoising_steps_d': self.ft_denoising_steps_d,
            'ft_denoising_steps_t': self.ft_denoising_steps_t,
            'min_sampling_denoising_std': self.min_sampling_denoising_std,
            'min_logprob_denoising_std': self.min_logprob_denoising_std,
            'eta': self.eta if hasattr(self, 'eta') else None,
            'learn_eta': self.learn_eta,
            'kwargs': {}  # You can include any additional arguments passed via kwargs
        })
        
        return config
    



    # ---------- Sampling ----------#

    def step(self):
        """
        Anneal min_sampling_denoising_std and fine-tuning denoising steps

        Current configs do not apply annealing
        """

        print("diffusion_vpg.py: VPGDiffusion.step()")

        # anneal min_sampling_denoising_std
        if type(self.min_sampling_denoising_std) is not float:
            self.min_sampling_denoising_std.step()

        # anneal denoising steps
        self.ft_denoising_steps_cnt += 1
        if (
            self.ft_denoising_steps_d > 0
            and self.ft_denoising_steps_t > 0
            and self.ft_denoising_steps_cnt % self.ft_denoising_steps_t == 0
        ):
            self.ft_denoising_steps = max(
                0, self.ft_denoising_steps - self.ft_denoising_steps_d
            )

            # update actor
            self.actor = self.actor_ft
            self.actor_ft = copy.deepcopy(self.actor)
            
            for param in self.actor.trainable_variables:
                param._trainable = False
            
            logging.info(
                f"Finished annealing fine-tuning denoising steps to {self.ft_denoising_steps}"
            )

    def get_min_sampling_denoising_std(self):

        print("diffusion_vpg.py: VPGDiffusion.get_min_sampling_denoising_std()")

        if type(self.min_sampling_denoising_std) is float:
            return self.min_sampling_denoising_std
        else:
            return self.min_sampling_denoising_std()

    # override
    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        use_base_policy=False,
        deterministic=False,
    ):

        print("diffusion_vpg.py: VPGDiffusion.p_mean_var()")

        noise = self.actor(x, t, cond=cond)
        if self.use_ddim:
            # ft_indices = torch.where(
            #     index >= (self.ddim_steps - self.ft_denoising_steps)
            # )[0]
            # ft_indices = torch_where(index >= (self.ddim_steps - self.ft_denoising_steps))[0]
            ft_indices = torch_where(index >= (self.ddim_steps - self.ft_denoising_steps))
            # [0]
            print("ft_indices = ", ft_indices)
            ft_indices = ft_indices[0]
            print("ft_indices = ", ft_indices)
        else:
            print("t = ", t)
            print("self.ft_denoising_steps = ", self.ft_denoising_steps)
            print("t < self.ft_denoising_steps = ", t < self.ft_denoising_steps)
            # ft_indices = torch.where(t < self.ft_denoising_steps)[0]
            ft_indices = torch_where(t < self.ft_denoising_steps)
            # [0]
            print("ft_indices = ", ft_indices)
            ft_indices = ft_indices[0]
            print("ft_indices = ", ft_indices)

        # Use base policy to query expert model, e.g. for imitation loss
        actor = self.actor if use_base_policy else self.actor_ft

        # overwrite noise for fine-tuning steps
        if len(ft_indices) > 0:
            cond_ft = {key: cond[key][ft_indices] for key in cond}
            noise_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / (tf.sqrt(alpha))
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:  # directly predicting x₀
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon = tf.clip_by_value(x_recon, -self.denoised_clip_value, self.denoised_clip_value)


            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - tf.sqrt(alpha) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            noise = tf.clip_by_value(noise, -self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε
            """
            if deterministic:    
                etas = tf.zeros((x.shape[0], 1, 1), dtype=x.dtype)
            else:
                etas = tf.expand_dims(self.eta(cond), axis=1)

            sigma = (
                etas * tf.sqrt( (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev) )
            )
            sigma = tf.clip_by_value(sigma, 1e-10, tf.float32.max)

            dir_xt_coef = tf.sqrt(tf.clip_by_value((1.0 - alpha_prev - sigma**2), 0, tf.float32.max))

            mu = (tf.sqrt(alpha_prev) * x_recon) + dir_xt_coef * noise
            var = tf.square(sigma)
            logvar = tf.math.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
            etas = tf.ones_like(mu).to(mu.device)  # always one for DDPM
        return mu, logvar, etas



    # override
    # @torch.no_grad()
    @tf.function
    def call(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        use_base_policy=False,
    ):
        """
        Forward pass for sampling actions.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            deterministic: If true, then std=0 with DDIM, or with DDPM, use normal schedule (instead of clipping at a higher value)
            return_chain: whether to return the entire chain of denoised actions
            use_base_policy: whether to use the frozen pre-trained policy instead
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
                chain: (B, K + 1, Ta, Da)
        """

        print("diffusion_vpg.py: VPGDiffusion.forward()")

        # device = self.betas.device
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = tf.shape(sample_data)[0]

        # Get updated minimum sampling denoising std
        min_sampling_denoising_std = self.get_min_sampling_denoising_std()

        # Loop
        # x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        x = tf.random.normal([B, self.horizon_steps, self.action_dim])

        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        chain = [] if return_chain else None
        if not self.use_ddim and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x)
        if self.use_ddim and self.ft_denoising_steps == self.ddim_steps:
            chain.append(x)

        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t)
            index_b = make_timesteps(B, i)
            mean, logvar, _ = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                use_base_policy=use_base_policy,
                deterministic=deterministic,
            )
            std = tf.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                if deterministic:
                    std = tf.zeros_like(std)
                else:
                    std = tf.clip_by_value(std, min_sampling_denoising_std, tf.float32.max)
            else:
                if deterministic and t == 0:
                    std = tf.zeros_like(std)
                elif deterministic:  # still keep the original noise
                    std = tf.clip_by_value(std, 1e-3, tf.float32.max)
                else:  # use higher minimum noise
                    std = tf.clip_by_value(std, min_sampling_denoising_std, tf.float32.max)

            noise = tf.clip_by_value(
                tf.random.normal(tf.shape(x)), -self.randn_clip_value, self.randn_clip_value
            )

            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = tf.clip_by_value(x, -self.final_action_clip_value, self.final_action_clip_value)

            if return_chain:
                if not self.use_ddim and t <= self.ft_denoising_steps:
                    chain.append(x)
                elif self.use_ddim and i >= (
                    self.ddim_steps - self.ft_denoising_steps - 1
                ):
                    chain.append(x)

        if return_chain:
            chain = tf.stack(chain, dim=1)
        return Sample(x, chain)

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        chains,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """
        Calculating the logprobs of the entire chain of denoised actions.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B x K, Ta, Da)
            entropy (if get_ent=True):  (B x K, Ta)
        """

        print("diffusion_vpg.py: VPGDiffusion.get_logprobs()")

        # Repeat cond for denoising_steps, flatten batch and time dimensions
        cond = {
            # key: cond[key]
            # .unsqueeze(1)
            # .repeat(1, self.ft_denoising_steps, *(1,) * (cond[key].ndim - 1))
            # .flatten(start_dim=0, end_dim=1)
            key: torch_flatten( tf.tile(
                tf.expand_dims(cond[key], axis=1),
                [1, self.ft_denoising_steps, *(1,) * (cond[key].ndim - 1)]
            ), 0, 1)
            
            for key in cond
        }  # less memory usage than einops?

        # Repeat t for batch dim, keep it 1-dim
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = torch_arange(
                start=self.ft_denoising_steps - 1,
                end=-1,
                step=-1,
                device=self.device,
            )
            # 4,3,2,1,0,4,3,2,1,0,...,4,3,2,1,0
        t_all = tf.tile(t_single, [chains.shape[0], 1])
        t_all = torch_flatten(t_all)

        if self.use_ddim:
            indices_single = torch_arange(
                start=self.ddim_steps - self.ft_denoising_steps,
                end=self.ddim_steps,
                device=self.device,
            )  # only used for DDIM

            indices = tf.tile(indices_single, chains.shape[0])

        else:
            indices = None

        # Split chains
        chains_prev = chains[:, :-1]
        chains_next = chains[:, 1:]

        # Flatten first two dimensions
        # chains_prev = chains_prev.reshape(-1, self.horizon_steps, self.action_dim)
        # chains_next = chains_next.reshape(-1, self.horizon_steps, self.action_dim)
        chains_prev = tf.reshape(chains_prev, [-1, self.horizon_steps, self.action_dim])
        chains_next = tf.reshape(chains_next, [-1, self.horizon_steps, self.action_dim])

        # Forward pass with previous chains
        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=indices,
            use_base_policy=use_base_policy,
        )
        std = tf.exp(0.5 * logvar)
        std = tf.clip_by_value(std, self.min_logprob_denoising_std, tf.float32.max)

        dist = Normal(next_mean, std)

        # Get logprobs with gaussian
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    def get_logprobs_subsample(
        self,
        cond,
        chains_prev,
        chains_next,
        denoising_inds,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """
        Calculating the logprobs of random samples of denoised chains.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B, Ta, Da)
            entropy (if get_ent=True):  (B, Ta)
            denoising_indices: (B, )
        """

        print("diffusion_vpg.py: VPGDiffusion.get_logprobs_subsample()")

        # Sample t for batch dim, keep it 1-dim
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = torch_arange(
                start=self.ft_denoising_steps - 1,
                end=-1,
                step=-1,
                device=self.device,
            )
            # 4,3,2,1,0,4,3,2,1,0,...,4,3,2,1,0
        t_all = t_single[denoising_inds]
        if self.use_ddim:
            ddim_indices_single = torch_arange(
                start=self.ddim_steps - self.ft_denoising_steps,
                end=self.ddim_steps,
                device=self.device,
            )  # only used for DDIM
            ddim_indices = ddim_indices_single[denoising_inds]
        else:
            ddim_indices = None

        # Forward pass with previous chains
        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=ddim_indices,
            use_base_policy=use_base_policy,
        )
        std = tf.exp(0.5 * logvar)
        std = tf.clip_by_value(std, self.min_logprob_denoising_std, tf.float32.max)

        dist = Normal(next_mean, std)

        # Get logprobs with gaussian
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob










    def loss_ori(self, cond, chains, reward):
        """
        REINFORCE loss. Not used right now.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            reward (to go): (b,)
        """

        print("diffusion_vpg.py: VPGDiffusion.loss()")

        # Get advantage
        # with tf.GradientTape() as tape:
        with torch_no_grad() as tape:
            value = tf.squeeze(self.critic(cond))  # (b,)

        # with torch.no_grad():
        #     value = self.critic(cond).squeeze()
        advantage = reward - value

        # Get logprobs for denoising steps from T-1 to 0
        logprobs, eta = self.get_logprobs(cond, chains, get_ent=True)
        # (n_steps x n_envs x K) x Ta x (Do+Da)

        # Ignore obs dimension, and then sum over action dimension
        logprobs = tf.reduce_sum(logprobs[:, :, :self.action_dim], axis=-1)
        # -> (n_steps x n_envs x K) x Ta

        # Reshape to group steps and environment
        logprobs = tf.reshape(logprobs, (-1, self.denoising_steps, self.horizon_steps))
        # -> (n_steps x n_envs) x K x Ta

        # Sum/avg over denoising steps
        logprobs = tf.reduce_mean(logprobs, axis=-2)  # -> (n_steps x n_envs) x Ta

        # Sum/avg over horizon steps
        logprobs = tf.reduce_mean(logprobs, axis=-1)  # -> (n_steps x n_envs)


        # Get REINFORCE loss
        loss_actor = tf.reduce_mean(-logprobs * advantage)

        # Train critic to predict state value
        pred = tf.squeeze(self.critic(cond))  # (b,)
        loss_critic = tf.reduce_mean(tf.square(pred - reward))

        return loss_actor, loss_critic, eta

