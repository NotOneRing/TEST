"""
Gaussian diffusion with DDPM and optionally DDIM sampling.

References:
Diffuser: https://github.com/jannerm/diffuser
Diffusion Policy: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py
Annotated DDIM/DDPM: https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

"""


from util.config import DEBUG
# DEBUG = True
# DEBUG = False


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
torch_flip, torch_randint

from util.torch_to_tf import torch_tensor_clone


# class DiffusionModel(tf.keras.layers.Layer):

from tensorflow.keras.saving import register_keras_serializable
@register_keras_serializable(package="Custom")
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

        # print("self.loss = ", self.loss)

        # print("self.loss() = ", self.loss())

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
        if not hasattr(self, "network"):
            self.network = network
        
        # self.network.batch_size = 

        self.network_path = network_path

        if self.network_path is not None:
            print("self.network_path is not None")
            # checkpoint = tf.train.Checkpoint(network=self.network)
            # checkpoint.restore(network_path)

            loadpath = network_path

            print("loadpath = ", loadpath)

            # # self.model.load_weights(loadpath)
            # # self.ema_model.load_weights(loadpath.replace(".h5", "_ema.h5"))

            # from keras.utils import get_custom_objects
            # from model.diffusion.mlp_diffusion import DiffusionMLP
            # # Register your custom class with Keras
            # get_custom_objects().update({
            #     'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
            #     'DiffusionMLP': DiffusionMLP,
            #     # 'VPGDiffusion': VPGDiffusion,
            #     # 'PPODiffusion': PPODiffusion
            # })


            # self.model = tf.keras.models.load_model(loadpath, custom_objects=get_custom_objects())
            # self.ema_model = tf.keras.models.load_model(loadpath.replace(".h5", "_ema.h5"), custom_objects=get_custom_objects())


            from keras.utils import get_custom_objects


            from model.diffusion.mlp_diffusion import DiffusionMLP
            from model.common.mlp import MLP, ResidualMLP
            from model.diffusion.modules import SinusoidalPosEmb
            from model.common.modules import SpatialEmb, RandomShiftsAug
            from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish


            # Register your custom class with Keras
            get_custom_objects().update({
                'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
                'DiffusionMLP': DiffusionMLP,
                # 'VPGDiffusion': VPGDiffusion,
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
                # 'Normal': Normal,
                # 'Layer': Layer,
                # 'Sample': Sample,
                # 'PPODiffusion': PPODiffusion
            })


            self.model = tf.keras.models.load_model(loadpath, custom_objects=get_custom_objects())
            self.ema_model = tf.keras.models.load_model(loadpath.replace(".h5", "_ema.h5"), custom_objects=get_custom_objects())



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

            self.ddim_alphas = (
                torch_tensor_clone(self.alphas_cumprod[self.ddim_t])
            )

            # self.ddim_alphas = tf.gather(self.alphas_cumprod, self.ddim_t)
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


            if DEBUG:
                print("DiffusionModel: __init__() DEBUG = True")

                print("DEBUG is True")
                self.loss_ori_t = None
                self.p_losses_noise = None
                self.call_noise = None
                self.call_noise = None
                self.call_x = None
                self.q_sample_noise = None
            else:
                print("DEBUG is False")


    def loss_ori(self
                 , training
                 , x_start, cond):
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

        if DEBUG:
            if self.loss_ori_t is None:
                self.loss_ori_t =  tf.cast( torch_randint(
                    low = 0, high = self.denoising_steps, size = (batch_size,)
                ), tf.int64)
                t = self.loss_ori_t
            else:
                t = self.loss_ori_t
        else:
            t =  tf.cast( torch_randint(
                low = 0, high = self.denoising_steps, size = (batch_size,)
            ), tf.int64)

        # t = tf.cast( torch_full((batch_size,), 3), tf.int64)  # 固定为 3


        # t = tf.fill([batch_size], 3)  # 固定为 3



        # Compute loss
        return self.p_losses(x_start, cond, t, 
                             training
                             )



    def p_losses(self, x_start, cond, t
                 , training
                 ):
        """
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||²

        Args:
            x_start: (batch_size, horizon_steps, action_dim)
            cond: dict with keys as step and value as observation
            t: batch of integers
        """
        print("diffusion.py: DiffusionModel.p_losses()")

        # # Forward process

        if DEBUG:
            if self.p_losses_noise is None:
                self.p_losses_noise = torch_randn_like(x_start)
                noise = self.p_losses_noise
            else:
                noise = self.p_losses_noise
        else:
            noise = torch_randn_like(x_start)

        # fixed_value = 1.0
        # noise = torch_full_like(x_start, fixed_value)  # 使用固定值替代随机噪声

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

        # # print("x_noisy.shape = ", x_noisy.shape)


        # # print("type(self.network) = ", type(self.network))

        # # print("self.network = ", self.network)

        # B, Ta, Da = x_noisy.shape

        # assert Ta == self.horizon_steps, "Ta != self.horizon_steps"
        # assert Da == self.action_dim, "Da != self.action_dim"

        # # flatten chunk
        # x_noisy = tf.reshape(x_noisy, [B, -1])

        # # flatten history
        # state = tf.reshape(cond["state"], [B, -1])

        # # print("t.shape = ", t.shape)

        # # append time and cond
        # time = tf.reshape(t, [B, 1])

        # # 提前展平 Batch * -1
        # # # # Predict
        # # x_recon = self.network(x_noisy, t, cond=cond, training=training_flag)



        # # Predict
        # x_recon = self.network(x_noisy, time, state, training=training_flag)


        print("self.network = ", self.network)

        # x_recon = self.network(x_noisy, t, cond = cond, training=training_flag)
        x_recon = self.network([x_noisy, t, cond["state"]]
                               , training=training)
                            #    )

        # summary = self.network.summary(x_noisy, t, cond["state"])
        # summary = self.network.summary(x_noisy, t, cond)

        # print("self.model.network.summary = ", summary)
        
        if self.predict_epsilon:
            return tf.reduce_mean(tf.square(x_recon - noise))  # Mean squared error
        else:
            return tf.reduce_mean(tf.square(x_recon - x_start))





    def p_mean_var(self, x, t, cond, index=None, network_override=None):

        print("diffusion.py: DiffusionModel.p_mean_var()", flush = True)

        if network_override is not None:
            noise = network_override(x, t, cond=cond)
        else:
            print("self.network = ", self.network)
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




    def q_sample(self, x_start, t, noise=None):
        """
        q(xₜ | x₀) = 𝒩(xₜ; √ α̅ₜ x₀, (1-α̅ₜ)I)
        xₜ = √ α̅ₜ xₒ + √ (1-α̅ₜ) ε
        """
        print("diffusion.py: DiffusionModel.q_sample()")

        # print("t = ", t)

        # print("extract function module:", extract.__module__)
        # print("extract function name:", extract.__name__)


        # Generate noise if not provided

        if DEBUG:
            if self.q_sample_noise is None:            
                if noise is None:
                    self.q_sample_noise = torch_randn_like(x_start)
                    noise = self.q_sample_noise
            else:
                if noise is None:
                    noise = self.q_sample_noise
        else:
            if noise is None:
                noise = torch_randn_like(x_start)


        # if noise is None:
        #     noise = torch_randn_like(x_start)

        print("DiffusionModel q_sample noise = ", noise)


        # print("self.sqrt_alphas_cumprod = ", self.sqrt_alphas_cumprod)
        # print("self.sqrt_one_minus_alphas_cumprod = ", self.sqrt_one_minus_alphas_cumprod)

        # print("x_start.shape = ", x_start.shape)
        # print("noise.shape = ", noise.shape)

        print("type(t) = ", type(t))

        # if isinstance(t, tf.keras.src.utils.tracking.TrackedDict):

        # from tensorflow.__internal__.tracking import TrackedDict

        # if not isinstance(t, tf.Tensor):
        #     t = dict(t)  # 转换为普通字典
        #     values = t['config']['value']
        #     dtype = t['config']['dtype']
        #     t = tf.convert_to_tensor(values, dtype=getattr(tf, dtype))


        print("DiffusionModel q_sample t = ", t)

        print("DiffusionModel q_sample type(t) = ", type(t) )



        # Compute x_t
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )






    # def forward(self, cond, deterministic=True):
    @tf.function
    def call(self, cond
            #  , deterministic=True
             ):
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
        if DEBUG:
            if self.call_x is None:
                self.call_x = torch_randn(B, self.horizon_steps, self.action_dim)
                x = self.call_x
            else:
                x = self.call_x
        else:
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
                    std = torch_clip(std, min=1e-3, max=tf.float32.max)

            # Sample noise and update `x`
            # noise = tf.random.normal(tf.shape(x))
            print("x.shape = ", x.shape)

            print("type(x.shape) = ", type(x.shape) )

            if DEBUG:
                if self.call_noise is None:            
                    self.call_noise = torch_randn_like( x  )
                    noise = self.call_noise
                else:
                    noise = self.call_noise
            else:
                noise = torch_randn_like( x  )

            torch_tensor_clamp_(noise, -self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            # Clamp action at the final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch_clamp(x, -self.final_action_clip_value, self.final_action_clip_value)

        # Return the result as a namedtuple
        return Sample(x, None)

        

























    def get_config(self):
        config = super(DiffusionModel, self).get_config()

        # config = {}

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


        from model.diffusion.mlp_diffusion import DiffusionMLP
        if isinstance( self.network, DiffusionMLP ):
            network_repr = self.network.get_config()
            print("network_repr = ", network_repr)
        else:
            print("type(self.network) = ", type(self.network))
            raise RuntimeError("not recognozed type of self.network")

        config.update({
            "network": network_repr,
            "horizon_steps": self.horizon_steps,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "network_path": self.network_path,
            "device": self.device,
            "denoised_clip_value": self.denoised_clip_value,
            "randn_clip_value": self.randn_clip_value,
            "final_action_clip_value": self.final_action_clip_value,
            "eps_clip_value": self.eps_clip_value,
            "denoising_steps": self.denoising_steps,
            "predict_epsilon": self.predict_epsilon,
            "use_ddim": self.use_ddim,
            "ddim_discretize": self.ddim_discretize,
            "ddim_steps": self.ddim_steps,

        })

        if DEBUG:
            print("DiffusionModel: get_config DEBUG = True")
            config.update({
            "loss_ori_t": self.loss_ori_t,
            "p_losses_noise": self.p_losses_noise,
            "call_noise": self.call_noise,
            "call_noise": self.call_noise,
            "call_x": self.call_x,
            "q_sample_noise": self.q_sample_noise,
            })
        

        print("DiffusionModel.config = ", config)
        
        return config


    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""

        from model.diffusion.mlp_diffusion import DiffusionMLP
        # from model.diffusion.diffusion import DiffusionModel
        from model.common.mlp import MLP, ResidualMLP, TwoLayerPreActivationResNetLinear
        from model.diffusion.modules import SinusoidalPosEmb
        from model.common.modules import SpatialEmb, RandomShiftsAug
        from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish, nn_Identity

        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            # 'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
            'DiffusionMLP': DiffusionMLP,
            # 'VPGDiffusion': VPGDiffusion,
            'SinusoidalPosEmb': SinusoidalPosEmb,  # 假设 SinusoidalPosEmb 是你自定义的层
            'MLP': MLP,                            # 自定义的 MLP 层
            'ResidualMLP': ResidualMLP,            # 自定义的 ResidualMLP 层
            'nn_Sequential': nn_Sequential,        # 自定义的 Sequential 类
            "nn_Identity": nn_Identity,
            'nn_Linear': nn_Linear,
            'nn_LayerNorm': nn_LayerNorm,
            'nn_Dropout': nn_Dropout,
            'nn_ReLU': nn_ReLU,
            'nn_Mish': nn_Mish,
            'SpatialEmb': SpatialEmb,
            'RandomShiftsAug': RandomShiftsAug,
            "TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
         }
        # Register your custom class with Keras
        get_custom_objects().update(cur_dict)

        # print('get_custom_objects() = ', get_custom_objects())

        network = config.pop("network")

        print("DiffusionModel from_config(): network = ", network)

        name = network["name"]
        print("name = ", name)

        # if name == "diffusion_mlp":
        #     name = "DiffusionMLP"
        if name.startswith("diffusion_mlp"):
            name = "DiffusionMLP"
            DiffusionMLP.from_config(network)
        else:
            raise RuntimeError("name not recognized")


        # if name in cur_dict:
        #     cur_dict[name].from_config(network)
        # else:
        #     raise RuntimeError("name not recognized")


        result = cls(network=network, **config)
        if DEBUG:

            # if not isinstance(config.pop("loss_ori_t"), tf.Tensor):
            loss_ori_t = config.pop("loss_ori_t")
            if loss_ori_t:
                print("Enter loss_ori_t")
                loss_ori_t = dict(loss_ori_t)  # 转换为普通字典
                values = loss_ori_t['config']['value']
                dtype = loss_ori_t['config']['dtype']
                loss_ori_t = tf.convert_to_tensor(values, dtype=getattr(tf, dtype))
                result.loss_ori_t = loss_ori_t
            else:
                result.loss_ori_t = None


            p_losses_noise = config.pop("p_losses_noise")
            if p_losses_noise:
                print("Enter p_losses_noise")
                p_losses_noise = dict(p_losses_noise)  # 转换为普通字典
                values = p_losses_noise['config']['value']
                dtype = p_losses_noise['config']['dtype']
                p_losses_noise = tf.convert_to_tensor(values, dtype=getattr(tf, dtype))
                result.p_losses_noise = p_losses_noise
            else:
                result.p_losses_noise = None


            call_noise = config.pop("call_noise")
            if call_noise:
                print("Enter call_noise")
                call_noise = dict()  # 转换为普通字典
                values = call_noise['config']['value']
                dtype = call_noise['config']['dtype']
                call_noise = tf.convert_to_tensor(values, dtype=getattr(tf, dtype))
                result.call_noise = call_noise
            else:
                result.call_noise = None


            call_x = config.pop("call_x")
            if call_x:
                print("Enter call_x")
                call_x = dict()  # 转换为普通字典
                values = call_x['config']['value']
                dtype = call_x['config']['dtype']
                call_x = tf.convert_to_tensor(values, dtype=getattr(tf, dtype))
                result.call_x = call_x
            else:
                result.call_x = None


            q_sample_noise = config.pop("q_sample_noise")
            if q_sample_noise:
                print("Enter q_sample_noise")
                q_sample_noise = dict(q_sample_noise)  # 转换为普通字典
                values = q_sample_noise['config']['value']
                dtype = q_sample_noise['config']['dtype']
                q_sample_noise = tf.convert_to_tensor(values, dtype=getattr(tf, dtype))
                result.q_sample_noise = q_sample_noise
            else:
                result.q_sample_noise = None


        
        return result

