"""
Gaussian diffusion with DDPM and optionally DDIM sampling.

References:
Diffuser: https://github.com/jannerm/diffuser
Diffusion Policy: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py
Annotated DDIM/DDPM: https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

"""

import logging

log = logging.getLogger(__name__)

from model.diffusion.sampling import (
    extract,
    cosine_beta_schedule,
    make_timesteps,
)

from collections import namedtuple

Sample = namedtuple("Sample", "trajectories chains")

import tensorflow as tf
import numpy as np



from util.torch_to_tf import torch_cumprod, torch_ones, torch_cat, torch_sqrt,\
torch_clamp, torch_log, torch_arange, torch_tensor_clamp_, torch_zeros_like, \
torch_clip, torch_exp, torch_randn_like, torch_randn, torch_full, torch_full_like, \
torch_flip


# class DiffusionModel(tf.keras.layers.Layer):
class DiffusionModel(tf.keras.Model):
    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        network_path=None,
        device="GPU:0",
        # Various clipping
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=None,
        eps_clip_value=None,  # DDIM only
        # DDPM parameters
        denoising_steps=100,
        predict_epsilon=True,
        # DDIM sampling
        use_ddim=False,
        ddim_discretize="uniform",
        ddim_steps=None,
        **kwargs,
    ):
        super(DiffusionModel, self).__init__()
        print("diffusion.py: DiffusionModel.__init__()")

        self.ddim_discretize = ddim_discretize
        
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        # Clip noise value at each denoising step
        self.denoised_clip_value = denoised_clip_value

        # Whether to clamp the final sampled action between [-1, 1]
        self.final_action_clip_value = final_action_clip_value

        # For each denoising step, we clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Clip epsilon for numerical stability
        self.eps_clip_value = eps_clip_value

        # print("before set up models")

        # Set up models
        self.network = network

        # self.network.batch_size = 

        self.network_path = network_path

        if self.network_path is not None:
            print("self.network_path is not None")
            checkpoint = tf.train.Checkpoint(network=self.network)
            checkpoint.restore(network_path)
            print(f"Loaded policy from {network_path}")

        print("after set up models")

        """
        DDPM parameters
        """
        self.betas = cosine_beta_schedule(denoising_steps)

        # print("self.betas = ", self.betas)
        # print("self.betas.shape = ", self.betas.shape)

        # print("after betas")

        self.alphas = 1.0 - self.betas

        # print("self.alphas = ", self.alphas)

        self.alphas_cumprod = torch_cumprod(self.alphas, dim=0)

        # print("self.alphas_cumprod = ", self.alphas_cumprod)
        
        # # 创建一个值为1的Tensor，数据类型和设备与 self.alphas_cumprod 相同
        # ones_tensor = tf.ones([1], dtype=self.alphas_cumprod.dtype)

        # # 将 self.alphas_cumprod 的第一个值从序列中移除
        # alphas_cumprod_truncated = self.alphas_cumprod[:-1]

        # # 将 ones_tensor 和 alphas_cumprod_truncated 进行拼接
        # self.alphas_cumprod_prev = tf.concat([ones_tensor, alphas_cumprod_truncated], axis=0)

        self.alphas_cumprod_prev = torch_cat(
            [torch_ones(1), self.alphas_cumprod[:-1]]
        )

        # print("self.alphas_cumprod_prev = ", self.alphas_cumprod_prev)

        # print("after alphas_cumprod_prev")





        """
        √ α̅ₜ
        """
        self.sqrt_alphas_cumprod = torch_sqrt(self.alphas_cumprod)
        """
        √ 1-α̅ₜ
        """
        self.sqrt_one_minus_alphas_cumprod = torch_sqrt(1.0 - self.alphas_cumprod)
        
        
        
        print("self.sqrt_alphas_cumprod = ", self.sqrt_alphas_cumprod)
        print("self.sqrt_one_minus_alphas_cumprod = ", self.sqrt_one_minus_alphas_cumprod)
        
        
        
        
        """
        √ 1\α̅ₜ
        """
        self.sqrt_recip_alphas_cumprod = torch_sqrt(1.0 / self.alphas_cumprod)
        """
        √ 1\α̅ₜ-1
        """
        self.sqrt_recipm1_alphas_cumprod = torch_sqrt(1.0 / self.alphas_cumprod - 1)
        """
        β̃ₜ = σₜ² = βₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
        """
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = torch_log(torch_clamp(self.ddpm_var, min=1e-20))
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        self.ddpm_mu_coef1 = (
            self.betas
            * torch_sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_mu_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch_sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )


        if use_ddim:

            print("after use_ddim")

            assert predict_epsilon, "DDIM requires predicting epsilon for now."
            if ddim_discretize == "uniform":
                step_ratio = self.denoising_steps // ddim_steps
                self.ddim_t = (
                    torch_arange(0, ddim_steps, device=self.device) * step_ratio
                )

                print("after ddim_discretize == uniform")

            else:
                raise ValueError("Unknown discretization method for DDIM.")

            print("after ddim_discretize")

            self.ddim_alphas = tf.gather(self.alphas_cumprod, self.ddim_t)
            self.ddim_alphas = tf.cast(self.ddim_alphas, tf.float32)

            self.ddim_alphas_sqrt = tf.sqrt(self.ddim_alphas)
            # self.ddim_alphas_prev = tf.concat(
            #     [tf.constant([1.0]), self.alphas_cumprod[:-1]], axis=0
            # )

            self.ddim_alphas_prev = torch_cat(
                [
                    tf.cast(tf.constant([1.0]), tf.float32),
                    self.alphas_cumprod[self.ddim_t[:-1]],
                ]
            )


            print("after ddim_alphas_prev")

            self.ddim_sqrt_one_minus_alphas = tf.sqrt(1.0 - self.ddim_alphas)

            ddim_eta = 0

            self.ddim_sigmas = ddim_eta * (
                (1 - self.ddim_alphas_prev)
                / (1 - self.ddim_alphas)
                * (1 - self.ddim_alphas / self.ddim_alphas_prev)
            ) ** 0.5

            print("after ddim_sigmas")

            # Flip all
            self.ddim_t = torch_flip(self.ddim_t, [0])
            
            self.ddim_alphas = torch_flip(self.ddim_alphas, [0])

            self.ddim_alphas_sqrt = torch_flip(self.ddim_alphas_sqrt, [0])
            self.ddim_alphas_prev = torch_flip(self.ddim_alphas_prev, [0])
            self.ddim_sqrt_one_minus_alphas = torch_flip(
                self.ddim_sqrt_one_minus_alphas, [0]
            )
            self.ddim_sigmas = torch_flip(self.ddim_sigmas, [0])







    def p_mean_var(self, x, t, cond, index=None, network_override=None):

        print("diffusion.py: DiffusionModel.p_mean_var()", flush = True)

        if network_override is not None:
            noise = network_override(x, t, cond=cond)
        else:
            noise = self.network(x, t, cond=cond)

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
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
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
            torch_tensor_clamp_(x_recon, -self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            torch_tensor_clamp_(noise, -self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε

            eta=0
            """
            sigma = extract(self.ddim_sigmas, index, x.shape)
            dir_xt = (1.0 - alpha_prev - sigma**2).sqrt() * noise
            mu = (alpha_prev**0.5) * x_recon + dir_xt
            var = sigma**2
            logvar = torch_log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
        return mu, logvar





    def get_config(self):
        config = super(DiffusionModel, self).get_config()

        print("get_config: diffusion.py: DiffusionModel.get_config()")

        # Debugging each attribute to make sure they are initialized correctly
        print(f"ddim_discretize: {self.ddim_discretize}")
        print(f"device: {self.device}")
        print(f"horizon_steps: {self.horizon_steps}")
        print(f"obs_dim: {self.obs_dim}")
        print(f"action_dim: {self.action_dim}")
        print(f"denoising_steps: {self.denoising_steps}")
        print(f"predict_epsilon: {self.predict_epsilon}")
        print(f"use_ddim: {self.use_ddim}")
        print(f"ddim_steps: {self.ddim_steps}")
        print(f"denoised_clip_value: {self.denoised_clip_value}")
        print(f"final_action_clip_value: {self.final_action_clip_value}")
        print(f"randn_clip_value: {self.randn_clip_value}")
        print(f"eps_clip_value: {self.eps_clip_value}")
        print(f"network: {self.network}")
        print(f"network_path: {self.network_path}")


        config.update({
            "ddim_discretize": self.ddim_discretize,
            "device": self.device,
            "horizon_steps": self.horizon_steps,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "denoising_steps": self.denoising_steps,
            "predict_epsilon": self.predict_epsilon,
            "use_ddim": self.use_ddim,
            "ddim_steps": self.ddim_steps,
            "denoised_clip_value": self.denoised_clip_value,
            "final_action_clip_value": self.final_action_clip_value,
            "randn_clip_value": self.randn_clip_value,
            "eps_clip_value": self.eps_clip_value,
            "network": self.network,
            "network_path": self.network_path,
        })
        return config





    # def forward(self, cond, deterministic=True):
    @tf.function
    def call(self, cond, deterministic=True):
        """
        Forward pass for sampling actions. Used in evaluating pre-trained/fine-tuned policy. Not modifying diffusion clipping.

        Args:
            cond: dict with keys state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
        """

        print("diffusion.py: DiffusionModel.forward()")

        # # Initialize
        # device = self.betas.device

        print("after device")

        sample_data = cond["state"] if "state" in cond else cond["rgb"]

        print("after sample_data")

        # B = tf.shape(sample_data)[0]
        # B = sample_data.get_shape().as_list()[0]
        B = sample_data.shape[0]



        print("B = ", B)

        print("self.horizon_steps = ", self.horizon_steps)

        print("self.action_dim = ", self.action_dim)

        # Starting random noise
        # x = tf.random.normal((B, self.horizon_steps, self.action_dim))
        x = torch_randn(B, self.horizon_steps, self.action_dim)

        # Define timesteps
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))

        # Main loop
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t)
            index_b = make_timesteps(B, i)

            # Compute mean and variance
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
            )
            std = torch_exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                std = torch_zeros_like(std)
            else:
                if t == 0:
                    std = torch_zeros_like(std)
                else:
                    std = torch_clip(std, clip_value_min=1e-3, clip_value_max=tf.float32.max)

            # Sample noise and update `x`
            # noise = tf.random.normal(tf.shape(x))
            noise = torch_randn_like(x.shape())
            torch_tensor_clamp_(noise, -self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            # Clamp action at the final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch_clamp(x, -self.final_action_clip_value, self.final_action_clip_value)

        # Return the result as a namedtuple
        return Sample(x, None)




    def loss(self, training_flag, x_start, cond):
        """
        Compute the loss for the given data and condition.

        Args:
            x_start: (batch_size, horizon_steps, action_dim)
            cond: dict with keys as step and value as observation

        Returns:
            loss: float
        """
        print("diffusion.py: DiffusionModel.loss()")

        # print("x_start = ", x_start)
        
        # print("cond = ", cond)


        # batch_size = tf.shape(x_start)[0]
        # batch_size = x_start.get_shape().as_list()[0]
        batch_size = x_start.shape[0]

        self.batch_size = batch_size
        self.network.batch_size = batch_size

        # print("tf.shape(x_start):", tf.shape(x_start))  # 返回形状
        # print("tf.shape(x_start)[0]:", tf.shape(x_start)[0])  # 直接获取第一个维度

        # print("int(batch_size.numpy()) = ", int(batch_size.numpy()))
        # print("int(batch_size) = ", int(batch_size))

        # batch_size = int(batch_size)

        print("batch_size = ", batch_size)

        # # 生成 [0, self.denoising_steps) 范围的随机整数
        t = tf.cast( torch_full((batch_size,), 3), tf.long)  # 固定为 3

        # t = tf.fill([batch_size], 3)  # 固定为 3



        # Compute loss
        return self.p_losses(x_start, cond, t, training_flag)




    def p_losses(self, x_start, cond, t, training_flag):
        """
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||²

        Args:
            x_start: (batch_size, horizon_steps, action_dim)
            cond: dict with keys as step and value as observation
            t: batch of integers
        """
        print("diffusion.py: DiffusionModel.p_losses()")

        # # Forward process
        # noise = tf.random.normal(tf.shape(x_start), dtype=x_start.dtype)
        fixed_value = 1.0
        noise = torch_full_like(x_start, fixed_value)  # 使用固定值替代随机噪声

        # # 假设 x_start 是一个已定义的张量
        # fixed_value = 1.0  # 固定数值
        # # noise = tf.fill(tf.shape(x_start), fixed_value)  # 使用 tf.fill 填充固定值
        # noise = tf.fill(x_start.shape, fixed_value)

        # print("x_start = ", x_start)
        
        # print("t = ", t)

        # print("noise = ", noise)

        # print("before q_sample")


        # print("type(self.network) = ", type(self.network))

        # print("self.network = ", self.network)


        # print("x_start.shape = ", x_start.shape)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # print("x_noisy.shape = ", x_noisy.shape)


        # print("type(self.network) = ", type(self.network))

        # print("self.network = ", self.network)

        B, Ta, Da = x_noisy.shape

        assert Ta == self.horizon_steps, "Ta != self.horizon_steps"
        assert Da == self.action_dim, "Da != self.action_dim"

        # flatten chunk
        x_noisy = tf.reshape(x_noisy, [B, -1])

        # flatten history
        state = tf.reshape(cond["state"], [B, -1])

        # print("t.shape = ", t.shape)

        # append time and cond
        time = tf.reshape(t, [B, 1])

        # 提前展平 Batch * -1
        # # # Predict
        # x_recon = self.network(x_noisy, t, cond=cond, training=training_flag)



        # Predict
        x_recon = self.network(x_noisy, time, state, training=training_flag)

        
        if self.predict_epsilon:
            return tf.reduce_mean(tf.square(x_recon - noise))  # Mean squared error
        else:
            return tf.reduce_mean(tf.square(x_recon - x_start))




    def q_sample(self, x_start, t, noise=None):
        """
        q(xₜ | x₀) = 𝒩(xₜ; √ α̅ₜ x₀, (1-α̅ₜ)I)
        xₜ = √ α̅ₜ xₒ + √ (1-α̅ₜ) ε
        """
        # print("diffusion.py: DiffusionModel.q_sample()")

        # print("t = ", t)

        # print("extract function module:", extract.__module__)
        # print("extract function name:", extract.__name__)


        # Generate noise if not provided
        if noise is None:
            # noise = tf.random.normal(shape=tf.shape(x_start), dtype=x_start.dtype)
            # noise = tf.random.normal(shape = x_start.shape, dtype=x_start.dtype)
            noise = torch_randn_like(x_start)

        # print("self.sqrt_alphas_cumprod = ", self.sqrt_alphas_cumprod)
        # print("self.sqrt_one_minus_alphas_cumprod = ", self.sqrt_one_minus_alphas_cumprod)

        # print("x_start.shape = ", x_start.shape)
        # print("noise.shape = ", noise.shape)

        # Compute x_t
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
















