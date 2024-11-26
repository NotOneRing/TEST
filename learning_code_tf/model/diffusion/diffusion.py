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

        print("before set up models")

        # Set up models
        self.network = network

        self.network_path = network_path

        if self.network_path is not None:
            checkpoint = tf.train.Checkpoint(network=self.network)
            checkpoint.restore(network_path)
            print(f"Loaded policy from {network_path}")

        print("after set up models")

        """
        DDPM parameters
        """
        self.betas = cosine_beta_schedule(denoising_steps)

        print("self.betas = ", self.betas)
        print("self.betas.shape = ", self.betas.shape)

        print("after betas")

        self.alphas = 1.0 - self.betas

        print("self.alphas = ", self.alphas)

        self.alphas_cumprod = tf.math.cumprod(self.alphas)

        print("self.alphas_cumprod = ", self.alphas_cumprod)
        
        # 创建一个值为1的Tensor，数据类型和设备与 self.alphas_cumprod 相同
        ones_tensor = tf.ones([1], dtype=self.alphas_cumprod.dtype)

        # 将 self.alphas_cumprod 的第一个值从序列中移除
        alphas_cumprod_truncated = self.alphas_cumprod[:-1]

        # 将 ones_tensor 和 alphas_cumprod_truncated 进行拼接
        self.alphas_cumprod_prev = tf.concat([ones_tensor, alphas_cumprod_truncated], axis=0)

        print("self.alphas_cumprod_prev = ", self.alphas_cumprod_prev)

        print("after alphas_cumprod_prev")










        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)
        
        
        print("self.sqrt_alphas_cumprod = ", self.sqrt_alphas_cumprod)
        print("self.sqrt_one_minus_alphas_cumprod = ", self.sqrt_one_minus_alphas_cumprod)
        
        
        
        
        self.sqrt_recip_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod - 1)

        print("before sqrt_recipm1_alphas_cumprod")

        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = tf.math.log(tf.clip_by_value(self.ddpm_var, 1e-20, 1e20))
        self.ddpm_mu_coef1 = (
            self.betas * tf.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_mu_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * tf.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        print("before ddpm_mu_coef2")

        if use_ddim:

            print("after use_ddim")

            assert predict_epsilon, "DDIM requires predicting epsilon for now."
            if ddim_discretize == "uniform":
                step_ratio = self.denoising_steps // ddim_steps
                self.ddim_t = tf.range(0, ddim_steps) * step_ratio

                print("after ddim_discretize == uniform")

            else:
                raise ValueError("Unknown discretization method for DDIM.")

            print("after ddim_discretize")

            self.ddim_alphas = tf.gather(self.alphas_cumprod, self.ddim_t)
            self.ddim_alphas_sqrt = tf.sqrt(self.ddim_alphas)
            self.ddim_alphas_prev = tf.concat(
                [tf.constant([1.0]), self.alphas_cumprod[:-1]], axis=0
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





    def get_config(self):
        config = super(DiffusionModel, self).get_config()

        super(DiffusionModel, self).__init__()
        print("diffusion.py: DiffusionModel.__init__()")

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



    # @tf.function
    # def forward(self, cond, deterministic=True):
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
        x = tf.random.normal((B, self.horizon_steps, self.action_dim))

        # Define timesteps
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = tf.range(self.denoising_steps - 1, -1, -1)

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
            std = tf.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                std = tf.zeros_like(std)
            else:
                if t == 0:
                    std = tf.zeros_like(std)
                else:
                    std = tf.clip_by_value(std, clip_value_min=1e-3, clip_value_max=tf.float32.max)

            # Sample noise and update `x`
            # noise = tf.random.normal(tf.shape(x))
            noise = tf.random.normal(x.shape())
            noise = tf.clip_by_value(noise, -self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            # Clamp action at the final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = tf.clip_by_value(x, -self.final_action_clip_value, self.final_action_clip_value)

        # Return the result as a namedtuple
        return Sample(x, None)


    # def _cosine_beta_schedule(self, denoising_steps):
    #     """
    #     Implements the cosine schedule for betas.
    #     """
    #     steps = tf.linspace(0, denoising_steps, denoising_steps)
    #     beta_schedule = 0.1 * (1 - tf.cos(steps / denoising_steps * np.pi))
    #     return beta_schedule

    def p_mean_var(self, x, t, cond):
        """
        Compute the mean and variance for the denoising process.

        Args:
            x: (batch_size, horizon_steps, action_dim)
            t: (batch_size,) - time steps
            cond: dict - condition inputs

        Returns:
            mean: predicted mean
            var: predicted variance
        """
        # Model prediction
        model_output = self.model(x, t, **cond)
        
        # Get parameters
        alphas_cumprod_t = tf.gather(self.alphas_cumprod, t)
        alphas_cumprod_prev = tf.gather(self.alphas_cumprod_prev, t)
        sqrt_recip_alphas_t = tf.gather(self.sqrt_recip_alphas, t)

        # Expand dimensions for broadcasting if necessary
        alphas_cumprod_t = tf.expand_dims(alphas_cumprod_t, axis=-1)
        alphas_cumprod_prev = tf.expand_dims(alphas_cumprod_prev, axis=-1)
        sqrt_recip_alphas_t = tf.expand_dims(sqrt_recip_alphas_t, axis=-1)

        # Mean prediction
        model_mean = sqrt_recip_alphas_t * (x - model_output)

        # Variance prediction
        var = 1 - alphas_cumprod_prev

        return model_mean, var





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

        print("x_start = ", x_start)
        
        print("cond = ", cond)


        # batch_size = tf.shape(x_start)[0]
        # batch_size = x_start.get_shape().as_list()[0]
        batch_size = x_start.shape[0]


        # print("tf.shape(x_start):", tf.shape(x_start))  # 返回形状
        # print("tf.shape(x_start)[0]:", tf.shape(x_start)[0])  # 直接获取第一个维度

        # print("int(batch_size.numpy()) = ", int(batch_size.numpy()))
        # print("int(batch_size) = ", int(batch_size))

        # batch_size = int(batch_size)

        print("batch_size = ", batch_size)

        # # 生成 [0, self.denoising_steps) 范围的随机整数
        # t = tf.random.uniform(
        #     shape=(batch_size,),
        #     minval=0,
        #     maxval=self.denoising_steps,
        #     dtype=tf.int32,
        # )

        t = tf.fill([batch_size], 3)  # 固定为 3



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

        # 假设 x_start 是一个已定义的张量
        fixed_value = 1.0  # 固定数值
        # noise = tf.fill(tf.shape(x_start), fixed_value)  # 使用 tf.fill 填充固定值
        noise = tf.fill(x_start.shape, fixed_value)
        print("x_start = ", x_start)
        
        print("t = ", t)

        print("noise = ", noise)

        print("before q_sample")


        print("type(self.network) = ", type(self.network))

        print("self.network = ", self.network)


        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)


        print("type(self.network) = ", type(self.network))

        print("self.network = ", self.network)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond, training=training_flag)

        
        if self.predict_epsilon:
            return tf.reduce_mean(tf.square(x_recon - noise))  # Mean squared error
        else:
            return tf.reduce_mean(tf.square(x_recon - x_start))




    def q_sample(self, x_start, t, noise=None):
        """
        q(xₜ | x₀) = 𝒩(xₜ; √ α̅ₜ x₀, (1-α̅ₜ)I)
        xₜ = √ α̅ₜ xₒ + √ (1-α̅ₜ) ε
        """
        print("diffusion.py: DiffusionModel.q_sample()")

        print("t = ", t)

        print("extract function module:", extract.__module__)
        print("extract function name:", extract.__name__)


        # Generate noise if not provided
        if noise is None:
            # noise = tf.random.normal(shape=tf.shape(x_start), dtype=x_start.dtype)
            noise = tf.random.normal(shape = x_start.shape, dtype=x_start.dtype)

        print("self.sqrt_alphas_cumprod = ", self.sqrt_alphas_cumprod)
        print("self.sqrt_one_minus_alphas_cumprod = ", self.sqrt_one_minus_alphas_cumprod)

        # Compute x_t
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
















