"""
Gaussian diffusion with DDPM and optionally DDIM sampling.

References:
Diffuser: https://github.com/jannerm/diffuser
Diffusion Policy: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_lowdim_policy.py
Annotated DDIM/DDPM: https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

"""


from util.config import DEBUG, NP_RANDOM


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
torch_clamp, torch_log, torch_arange, torch_zeros_like, \
torch_clip, torch_exp, torch_randn_like, torch_randn, torch_full, torch_full_like, \
torch_flip, torch_randint, torch_ones_like, torch_no_grad, torch_clamp

from util.torch_to_tf import torch_tensor_clone


from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, OUTPUT_POSITIONS, OUTPUT_FUNCTION_HEADER, NP_RANDOM, METHOD_NAME


from util.torch_to_tf import nn_Parameter, torch_tensor





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




        if DEBUG or NP_RANDOM:
            if OUTPUT_POSITIONS:
                print("DiffusionModel: __init__() DEBUG = True")

                print("DEBUG is True")
            self.loss_ori_t = None
            self.p_losses_noise = None
            self.call_noise = None
            self.call_noise = None
            self.call_x = None
            self.q_sample_noise = None
        else:
            if OUTPUT_POSITIONS:
                print("DEBUG is False")




        self.env_name = kwargs.get("env_name", None)


        print("self.env_name = ", self.env_name)
        
        super().__init__()

        if OUTPUT_FUNCTION_HEADER:
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


        print("network = ", network)
        print("type(network) = ", type(network))


        # Set up models
        if not hasattr(self, "network"):
            self.network = network
        

        self.network_path = network_path



        if OUTPUT_POSITIONS:
            print("after set up models")

        """
        DDPM parameters
        """
        self.betas = cosine_beta_schedule(denoising_steps)


        self.alphas = 1.0 - self.betas


        self.alphas_cumprod = torch_cumprod(self.alphas, dim=0)

        # # create a Tensor of value 1, with the same data type and device as self.alphas_cumprod
        # # remove the first value from the sequence of self.alphas_cumprod
        # # concatenate ones_tensor and alphas_cumprod_truncated
        self.alphas_cumprod_prev = torch_cat(
            [torch_ones(1), self.alphas_cumprod[:-1]]
        )






        """
        √ α̅ₜ
        """
        self.sqrt_alphas_cumprod = torch_sqrt(self.alphas_cumprod)
        """
        √ 1-α̅ₜ
        """
        self.sqrt_one_minus_alphas_cumprod = torch_sqrt(1.0 - self.alphas_cumprod)
        
        
        
        if OUTPUT_VARIABLES:            
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

            if OUTPUT_POSITIONS:
                print("after use_ddim")

            assert predict_epsilon, "DDIM requires predicting epsilon for now."
            if ddim_discretize == "uniform":
                step_ratio = self.denoising_steps // ddim_steps
                self.ddim_t = (
                    torch_arange(0, ddim_steps, device=self.device) * step_ratio
                )

                if OUTPUT_POSITIONS:
                    print("after ddim_discretize == uniform")

            else:
                raise ValueError("Unknown discretization method for DDIM.")

            if OUTPUT_POSITIONS:
                print("after ddim_discretize")



            print("Diffusion: self.ddim_t = ", self.ddim_t)
            print("Diffusion: type(self.ddim_t) = ", type(self.ddim_t))


            temp_result = tf.gather(self.alphas_cumprod, self.ddim_t, axis=0)            


            self.ddim_alphas = (
                torch_tensor_clone(temp_result)
            )

            self.ddim_alphas = tf.cast(self.ddim_alphas, tf.float32)

            self.ddim_alphas_sqrt = tf.sqrt(self.ddim_alphas)


            self.ddim_alphas_prev = torch_cat(
                [
                    tf.cast(tf.constant([1.0]), tf.float32),
                    tf.gather(self.alphas_cumprod, self.ddim_t[:-1], axis=0)
                ]
            )


            if OUTPUT_POSITIONS:
                print("after ddim_alphas_prev")

            self.ddim_sqrt_one_minus_alphas = tf.sqrt(1.0 - self.ddim_alphas)

            ddim_eta = 0

            self.ddim_sigmas = ddim_eta * (
                (1 - self.ddim_alphas_prev)
                / (1 - self.ddim_alphas)
                * (1 - self.ddim_alphas / self.ddim_alphas_prev)
            ) ** 0.5

            if OUTPUT_POSITIONS:
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




        if self.network_path is not None:
            print("self.network_path is not None")
            loadpath = network_path

            print("loadpath = ", loadpath)

            if loadpath.endswith(".h5") or loadpath.endswith(".keras"):
                print('loadpath.endswith(".h5") or loadpath.endswith(".keras")')
            else:
                loadpath = network_path.replace('.pt', '.keras')

            from model.diffusion.mlp_diffusion import DiffusionMLP
            from model.diffusion.diffusion import DiffusionModel
            from model.common.mlp import MLP, ResidualMLP, TwoLayerPreActivationResNetLinear
            from model.diffusion.modules import SinusoidalPosEmb
            from model.common.modules import SpatialEmb, RandomShiftsAug
            from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish, nn_Identity, nn_Conv1d

            from model.diffusion.unet import Downsample1d, ResidualBlock1D, Conv1dBlock, Unet1D

            from tensorflow.keras.utils import get_custom_objects

            cur_dict = {
                'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
                'DiffusionMLP': DiffusionMLP,
                'SinusoidalPosEmb': SinusoidalPosEmb,   
                'MLP': MLP,                            # Custom MLP layer
                'ResidualMLP': ResidualMLP,            # Custom ResidualMLP layer
                'nn_Sequential': nn_Sequential,        # Custom Sequential class
                "nn_Identity": nn_Identity,
                'nn_Linear': nn_Linear,
                'nn_LayerNorm': nn_LayerNorm,
                'nn_Dropout': nn_Dropout,
                'nn_ReLU': nn_ReLU,
                'nn_Mish': nn_Mish,
                'SpatialEmb': SpatialEmb,
                'RandomShiftsAug': RandomShiftsAug,
                "TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
                'Downsample1d': Downsample1d,
                'ResidualBlock1D':ResidualBlock1D,
                'Conv1dBlock': Conv1dBlock,
                'nn_Conv1d': nn_Conv1d,
                'Unet1D': Unet1D,
            }
            # Register custom class with Keras
            get_custom_objects().update(cur_dict)


            self.network = tf.keras.models.load_model( loadpath.replace(".keras", "_network.keras") ,  custom_objects=get_custom_objects() )


            if OUTPUT_VARIABLES:
                self.output_weights(self.network)

            # self.build_actor(self.network)
            if "ViT" in METHOD_NAME:            
                self.build_actor_vision(self.network)
            else:
                self.build_actor(self.network)



            print("DiffusionModel: self.network = ", self.network )





    def get_config(self):
        config = super(DiffusionModel, self).get_config()

        if OUTPUT_FUNCTION_HEADER:
            print("get_config: diffusion.py: DiffusionModel.get_config()")

        if OUTPUT_VARIABLES:
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


        from model.diffusion.mlp_diffusion import DiffusionMLP, VisionDiffusionMLP
        from model.diffusion.unet import Unet1D

        if isinstance( self.network, (DiffusionMLP, Unet1D, VisionDiffusionMLP) ):
            network_repr = self.network.get_config()
            if OUTPUT_VARIABLES:
                print("network_repr = ", network_repr)
        else:
            if OUTPUT_VARIABLES:
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



        if hasattr(self, "env_name"):
            print("get_config(): self.env_name = ", self.env_name)
            config.update({
            "env_name": self.env_name,
            })
        else:
            print("get_config(): self.env_name = ", None)
        



        if OUTPUT_VARIABLES:
            print("DiffusionModel.config = ", config)
        
        return config


    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""

        from model.diffusion.mlp_diffusion import DiffusionMLP, VisionDiffusionMLP
        # from model.diffusion.diffusion import DiffusionModel
        from model.common.mlp import MLP, ResidualMLP, TwoLayerPreActivationResNetLinear
        from model.diffusion.modules import SinusoidalPosEmb
        from model.common.modules import SpatialEmb, RandomShiftsAug
        from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, \
            nn_Dropout, nn_ReLU, nn_Mish, nn_Identity, nn_Conv1d, nn_ConvTranspose1d

        from model.diffusion.unet import Unet1D, ResidualBlock1D


        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'DiffusionMLP': DiffusionMLP,
            'SinusoidalPosEmb': SinusoidalPosEmb,   
            'MLP': MLP,                            # Custom MLP layer
            'ResidualMLP': ResidualMLP,            # Custom ResidualMLP layer
            'nn_Sequential': nn_Sequential,        # Custom Sequential class
            "nn_Identity": nn_Identity,
            'nn_Linear': nn_Linear,
            'nn_LayerNorm': nn_LayerNorm,
            'nn_Dropout': nn_Dropout,
            'nn_ReLU': nn_ReLU,
            'nn_Mish': nn_Mish,
            'SpatialEmb': SpatialEmb,
            'RandomShiftsAug': RandomShiftsAug,
            "TwoLayerPreActivationResNetLinear": TwoLayerPreActivationResNetLinear,
            "Unet1D": Unet1D,
            "ResidualBlock1D": ResidualBlock1D,
            "nn_Conv1d": nn_Conv1d,
            "nn_ConvTranspose1d": nn_ConvTranspose1d
         }
        # Register custom class with Keras
        get_custom_objects().update(cur_dict)

        network = config.pop("network")

        if OUTPUT_VARIABLES:
            print("DiffusionModel from_config(): network = ", network)

        name = network["name"]
    
        # if OUTPUT_VARIABLES:
        print("name = ", name)


        if name.startswith("diffusion_mlp"):
            name = "DiffusionMLP"
            network = DiffusionMLP.from_config(network)
        elif name.startswith("unet1d"):
            network = Unet1D.from_config(network)
        elif name.startswith("vision_diffusion_mlp"):
            network = VisionDiffusionMLP.from_config(network)
        else:
            raise RuntimeError("name not recognized")




        result = cls(
            network=network, 
            **config)



        env_name = config.pop("env_name")
        if env_name:
            if OUTPUT_POSITIONS:
                print("Enter env_name")
            result.env_name = env_name
        else:
            result.env_name = None

        return result





            








    def loss_ori(self
                 , training,
                x, *args):
        """
        Compute the loss for the given data and condition.

        Args:
            x_start: (batch_size, horizon_steps, action_dim)
            cond: dict with keys as step and value as observation

        Returns:
            loss: float
        """

        if OUTPUT_FUNCTION_HEADER:
            print("diffusion.py: DiffusionModel.loss()")


        batch_size = x.shape[0]

        self.batch_size = batch_size
        self.network.batch_size = batch_size


        if OUTPUT_VARIABLES:
            print("batch_size = ", batch_size)


        if DEBUG or NP_RANDOM:
            if self.loss_ori_t is None or training:

                self.loss_ori_t =  tf.cast( tf.convert_to_tensor(np.random.randint( 0, self.denoising_steps, (batch_size,) ) ), tf.int64 )

                t = self.loss_ori_t
            else:
                t = self.loss_ori_t


        else:
            t =  tf.cast( torch_randint(
                low = 0, high = self.denoising_steps, size = (batch_size,)
            ), tf.int64)

        return self.p_losses(x, *args,  t, training)























    def loss_ori_build(self,
                network
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

        if OUTPUT_FUNCTION_HEADER:
            print("diffusion.py: DiffusionModel.loss_ori_build()")

        batch_size = x_start.shape[0]

        self.batch_size = batch_size
        network.batch_size = batch_size

        if OUTPUT_VARIABLES:
            print("batch_size = ", batch_size)


        if DEBUG or NP_RANDOM:
            if self.loss_ori_t is None or training:

                self.loss_ori_t =  tf.cast( tf.convert_to_tensor(np.random.randint( 0, self.denoising_steps, (batch_size,) )), tf.int64)

                t = self.loss_ori_t
            else:
                t = self.loss_ori_t

        else:
            t =  tf.cast( torch_randint(
                low = 0, high = self.denoising_steps, size = (batch_size,)
            ), tf.int64)


        return DiffusionModel.p_losses_build(self, network, x_start, cond, t, training )
























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

        if OUTPUT_FUNCTION_HEADER:
            print("diffusion.py: DiffusionModel.p_losses()")

        # # Forward process

        if DEBUG or NP_RANDOM:
            if self.p_losses_noise is None or training:
                self.p_losses_noise = tf.convert_to_tensor( np.random.randn( *(x_start.numpy().shape) ), dtype=tf.float32 )

                noise = self.p_losses_noise
            else:
                noise = self.p_losses_noise

        else:
            noise = torch_randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, training=training)

        # if OUTPUT_VARIABLES:
        print("self.network = ", self.network)

        if 'rgb' in cond:
            x_recon = self.network([x_noisy, t, cond["state"], cond["rgb"]]
                                , training=training)
        else:
            x_recon = self.network([x_noisy, t, cond["state"]]
                                , training=training)

        
        if self.predict_epsilon:
            return tf.reduce_mean(tf.square(x_recon - noise))  # Mean squared error
        else:
            return tf.reduce_mean(tf.square(x_recon - x_start))































    def p_losses_build(self, network, x_start, cond, t
                 , training
                 ):
        """
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||²

        Args:
            x_start: (batch_size, horizon_steps, action_dim)
            cond: dict with keys as step and value as observation
            t: batch of integers
        """

        if OUTPUT_FUNCTION_HEADER:
            print("diffusion.py: DiffusionModel.p_losses_build()")


        if DEBUG or NP_RANDOM:
            if self.p_losses_noise is None or training:
                self.p_losses_noise = tf.convert_to_tensor( np.random.randn( *(x_start.numpy().shape) ), dtype=tf.float32 )

                noise = self.p_losses_noise
            else:
                noise = self.p_losses_noise

        else:
            noise = torch_randn_like(x_start)

        
        x_noisy = DiffusionModel.q_sample(self, x_start=x_start, t=t, noise=noise, training=training)



        if OUTPUT_VARIABLES:
            print("self.network = ", self.network)


        if 'rgb' in cond:
            x_recon = network([x_noisy, t, cond["state"], cond["rgb"]], training=training)
                                #    )
        else:
            x_recon = network([x_noisy, t, cond["state"]], training=training)
                                #    )


        if self.predict_epsilon:
            return tf.reduce_mean(tf.square(x_recon - noise))  # Mean squared error
        else:
            return tf.reduce_mean(tf.square(x_recon - x_start))



























    def p_mean_var(self, x, t, cond_state, index=None, network_override=None):

        if OUTPUT_FUNCTION_HEADER:
            print("diffusion.py: DiffusionModel.p_mean_var()", flush = True)

        if network_override is not None:
            noise = network_override([x, t, cond_state])
        else:
            if OUTPUT_VARIABLES:
                print("self.network = ", self.network)
            noise = self.network([x, t, cond_state])

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

                if OUTPUT_VARIABLES:
                    print("self.sqrt_recip_alphas_cumprod = ", self.sqrt_recip_alphas_cumprod)
                    print("t = ", t)
                    print("x.shape = ", x.shape)

                extract_result1 = extract(self.sqrt_recip_alphas_cumprod, t, x.shape)

                if OUTPUT_VARIABLES:
                    print("extract_result1 = ", extract_result1)

                    print("x.dtype = ", x.dtype)
                    print("extract_result1.dtype = ", extract_result1.dtype)

                x_recon = (
                    extract_result1 * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:  # directly predicting x₀
            x_recon = noise

    
        if OUTPUT_VARIABLES:
            print("DiffusionModel: p_mean_var(): x_recon = ", x_recon)



        if self.denoised_clip_value is not None:
            x_recon = torch_clamp(x_recon, -self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            noise = torch_clamp(noise, -self.eps_clip_value, self.eps_clip_value)

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






    def q_sample(self, x_start, t, noise=None, training=True):
        """
        q(xₜ | x₀) = 𝒩(xₜ; √ α̅ₜ x₀, (1-α̅ₜ)I)
        xₜ = √ α̅ₜ xₒ + √ (1-α̅ₜ) ε
        """

        if OUTPUT_FUNCTION_HEADER:
            print("diffusion.py: DiffusionModel.q_sample()")


        # Generate noise if not provided
        if DEBUG or NP_RANDOM:
            if self.q_sample_noise is None or training: 

                if noise is None:

                    self.q_sample_noise = tf.convert_to_tensor( np.random.randn(*(x_start.numpy().shape)), dtype=tf.float32)
                    
                    noise = self.q_sample_noise

            else:

                if noise is None:
                    noise = self.q_sample_noise

        else:

            if noise is None:

                noise = torch_randn_like(x_start)


        if OUTPUT_VARIABLES:
            print("Diffusion: q_sample(): noise = ", noise)



        if OUTPUT_VARIABLES:
            print("DiffusionModel q_sample noise = ", noise)


        if OUTPUT_VARIABLES:
            print("type(t) = ", type(t))


        if OUTPUT_VARIABLES:
            print("DiffusionModel q_sample t = ", t)

            print("DiffusionModel q_sample type(t) = ", type(t) )


        extract1 = extract(self.sqrt_alphas_cumprod, t, x_start.shape)


        extract2 = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)


        return (
            extract1 * x_start
            + extract2 * noise
        )






    @tf.function
    def call(self, 
             cond_state,
            training=True
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

        with torch_no_grad() as tape:

            if OUTPUT_FUNCTION_HEADER:
                print("diffusion.py: DiffusionModel.forward()")


            if OUTPUT_POSITIONS:
                print("after device")

            sample_data = cond_state

            if OUTPUT_POSITIONS:
                print("after sample_data")

            B = tf.shape(sample_data)[0] 

            if OUTPUT_VARIABLES:
                print("B = ", B)
                print("self.horizon_steps = ", self.horizon_steps)
                print("self.action_dim = ", self.action_dim)

            if OUTPUT_VARIABLES:
                print("B = ", B)

                print("self.horizon_steps = ", self.horizon_steps)

                print("self.action_dim = ", self.action_dim)


            if DEBUG or NP_RANDOM:
                if self.call_x is None or training:

                    self.call_x = tf.convert_to_tensor( np.random.randn(B, self.horizon_steps, self.action_dim), dtype=tf.float32)

                    x = self.call_x

                    if OUTPUT_VARIABLES:
                        print("x from DEBUG branch")
                else:
                    x = self.call_x
            else:
                x = torch_randn(B, self.horizon_steps, self.action_dim)

            if OUTPUT_VARIABLES:
                print("Diffusion.call(): x1 = ", x)

            # Define timesteps
            if self.use_ddim:
                t_all = self.ddim_t
            else:
                t_all = list(reversed(range(self.denoising_steps)))

            if OUTPUT_VARIABLES:
                print("Diffusion.call(): t_all = ", t_all)

            # Main loop
            for i, t in enumerate(t_all):
                t_b = make_timesteps(B, t)
                index_b = make_timesteps(B, i)

                # Compute mean and variance
                mean, logvar = self.p_mean_var(
                    x=x,
                    t=t_b,
                    cond_state=cond_state,
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

                if OUTPUT_VARIABLES:
                    print("x.shape = ", x.shape)

                    print("type(x.shape) = ", type(x.shape) )

                if DEBUG or NP_RANDOM:
                    noise = tf.Variable( np.random.randn( *(x.numpy().shape) ) , dtype=tf.float32 )
                else:
                    noise = torch_randn_like( x  )

                if OUTPUT_VARIABLES:
                    print("Diffusion.call(): std = ", std)

                    print("Diffusion.call(): noise = ", noise)

                noise = torch_clamp(noise, -self.randn_clip_value, self.randn_clip_value)
                x = mean + std * noise

                if OUTPUT_VARIABLES:
                    print("Diffusion.call(): x2 = ", x)

                # Clamp action at the final step
                if self.final_action_clip_value is not None and i == len(t_all) - 1:
                    x = torch_clamp(x, -self.final_action_clip_value, self.final_action_clip_value)

                    if OUTPUT_VARIABLES:
                        print("Diffusion.call(): x3 = ", x)

            # Return the result as a namedtuple
            return Sample(x, None)

            






    def build_actor(self, actor, shape1=None, shape2=None):
        # return
    
        print("build_actor: self.env_name = ", self.env_name)

        if shape1 != None and shape2 != None:
            pass
        # Gym - hopper/walker2d/halfcheetah
        elif self.env_name == "hopper-medium-v2":
            shape1 = (128, 4, 3)
            shape2 = (128, 1, 11)
        elif self.env_name == "kitchen-complete-v0":
            shape1 = (128, 4, 9)
            shape2 = (128, 1, 60)
        elif self.env_name == "kitchen-mixed-v0":
            shape1 = (256, 4, 9)
            shape2 = (256, 1, 60)
        elif self.env_name == "kitchen-partial-v0":
            shape1 = (128, 4, 9)
            shape2 = (128, 1, 60)
        elif self.env_name == "walker2d-medium-v2":
            shape1 = (128, 4, 6)
            shape2 = (128, 1, 17)
        elif self.env_name == "halfcheetah-medium-v2":
            shape1 = (128, 4, 6)
            shape2 = (128, 1, 17)
        # Robomimic - lift/can/square/transport
        elif self.env_name == "lift":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 19)

        elif self.env_name == "can":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 23)

        elif self.env_name == "square":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 23)

        elif self.env_name == "transport":
            shape1 = (256, 8, 14)
            shape2 = (256, 1, 59)

        # the same name "avoiding-m5" for D3IL with avoid_m1/m2/m3
        elif self.env_name == "avoiding-m5" or self.env_name == "avoid":
            shape1 = (16, 4, 2)
            shape2 = (16, 1, 4)

        # Furniture-Bench - one_leg/lamp/round_table_low/med
        elif self.env_name == "lamp_low_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 44)
        elif self.env_name == "lamp_med_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 44)
        elif self.env_name == "one_leg_low_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 58)
        elif self.env_name == "one_leg_med_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 58)
        elif self.env_name == "round_table_low_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 44)
        elif self.env_name == "round_table_med_dim":
            shape1 = (256, 8, 10)
            shape2 = (256, 1, 44)
        
        else:
            raise RuntimeError("The build shape is not implemented for current dataset")



        if OUTPUT_VARIABLES:
            print("type(shape1) = ", type(shape1))
            print("type(shape2) = ", type(shape2))

            print("shape1 = ", shape1)
            print("shape2 = ", shape2)


        param1 = torch_ones(*shape1)
        param2 = torch_ones(*shape2)

        build_dict = {'state': param2}


        all_one_build_result = self.loss_ori_build(actor, training=False, x_start = param1, cond=build_dict)

        print("all_one_build_result = ", all_one_build_result)
















    def build_actor_vision(self, actor, shape1=None, shape2=None):
    
        print("build_actor_vision: self.env_name = ", self.env_name)

        if shape1 != None and shape2 != None:
            pass

        elif self.env_name == "square":
            shape1 = (256, 4, 7)
            shape2 = (256, 1, 9)
            shape3 = (256, 1, 3, 96, 96)     
        elif self.env_name == "transport":
            shape1 =  (256, 8, 14)
            shape2 =  (256, 1, 18)
            shape3 =  (256, 1, 6, 96, 96)
        else:
            raise RuntimeError("The build shape is not implemented for current dataset")


        param1 = torch_ones(*shape1)
        param2 = torch_ones(*shape2)
        param3 = torch_ones(*shape3)

        build_dict = {'state': param2}
        build_dict['rgb'] = param3


        all_one_build_result = self.loss_ori_build(actor, training=False, x_start = param1, cond=build_dict)

        print("all_one_build_result = ", all_one_build_result)













    def load_pickle(self, network_path):
        pkl_file_path = network_path.replace('.pt', '_ema.pkl')

        print("pkl_file_path = ", pkl_file_path)

        import pickle
        with open(pkl_file_path, 'rb') as file:
            params_dict = pickle.load(file)

        if OUTPUT_VARIABLES:
            print("params_dict = ", params_dict)


        if OUTPUT_VARIABLES:
            print("before self.network.time_embedding[1].trainable_weights[0].assign(params_dict['network.time_embedding.1.weight'].T)")
        self.network.time_embedding[1].trainable_weights[0].assign(params_dict['network.time_embedding.1.weight'].T)  # kernel
        if OUTPUT_VARIABLES:
            print("before self.network.time_embedding[1].trainable_weights[1].assign(params_dict['network.time_embedding.1.bias'])")
        self.network.time_embedding[1].trainable_weights[1].assign(params_dict['network.time_embedding.1.bias'])     # bias

        if OUTPUT_VARIABLES:
            print("before self.network.time_embedding[3].trainable_weights[0].assign(params_dict['network.time_embedding.3.weight'].T)")
        self.network.time_embedding[3].trainable_weights[0].assign(params_dict['network.time_embedding.3.weight'].T)  # kernel
        if OUTPUT_VARIABLES:
            print("before self.network.time_embedding[3].trainable_weights[1].assign(params_dict['network.time_embedding.3.bias'])")
        self.network.time_embedding[3].trainable_weights[1].assign(params_dict['network.time_embedding.3.bias'])     # bias


        if 'network.cond_mlp.moduleList.0.linear_1.weight' in params_dict:
            self.network.cond_mlp.moduleList[0].trainable_weights[0].assign(params_dict['network.cond_mlp.moduleList.0.linear_1.weight'].T)  # kernel

        if 'network.cond_mlp.moduleList.0.linear_1.bias' in params_dict:
            self.network.cond_mlp.moduleList[0].trainable_weights[1].assign(params_dict['network.cond_mlp.moduleList.0.linear_1.bias'])  # kernel

        if 'network.cond_mlp.moduleList.1.linear_1.weight' in params_dict:
            self.network.cond_mlp.moduleList[1].trainable_weights[0].assign(params_dict['network.cond_mlp.moduleList.1.linear_1.weight'].T)  # kernel

        if 'network.cond_mlp.moduleList.1.linear_1.bias' in params_dict:
            self.network.cond_mlp.moduleList[1].trainable_weights[1].assign(params_dict['network.cond_mlp.moduleList.1.linear_1.bias'])  # kernel




        if OUTPUT_VARIABLES:
            print("before self.network.mlp_mean.my_layers[0].trainable_weights[0].assign(params_dict['network.mlp_mean.layers.0.weight'].T)")
        self.network.mlp_mean.my_layers[0].trainable_weights[0].assign(params_dict['network.mlp_mean.layers.0.weight'].T)  # kernel
        if OUTPUT_VARIABLES:
            print("before self.network.mlp_mean.my_layers[0].trainable_weights[1].assign(params_dict['network.mlp_mean.layers.0.bias'])")
        self.network.mlp_mean.my_layers[0].trainable_weights[1].assign(params_dict['network.mlp_mean.layers.0.bias'])     # bias


        if OUTPUT_VARIABLES:
            print("before self.network.mlp_mean.my_layers[1].l1.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.1.l1.weight'].T)")
        self.network.mlp_mean.my_layers[1].l1.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.1.l1.weight'].T)  # kernel
        if OUTPUT_VARIABLES:
            print("before self.network.mlp_mean.my_layers[1].l1.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.1.l1.bias'])")
        self.network.mlp_mean.my_layers[1].l1.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.1.l1.bias'])     # bias
        if OUTPUT_VARIABLES:
            print("before self.network.mlp_mean.my_layers[1].l2.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.1.l2.weight'].T)")
        self.network.mlp_mean.my_layers[1].l2.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.1.l2.weight'].T)  # kernel
        if OUTPUT_VARIABLES:
            print("before self.network.mlp_mean.my_layers[1].l2.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.1.l2.bias'])")
        self.network.mlp_mean.my_layers[1].l2.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.1.l2.bias'])     # bias


        if OUTPUT_VARIABLES:
            print("before self.network.mlp_mean.my_layers[2].trainable_weights[0].assign(params_dict['network.mlp_mean.layers.2.weight'].T)")
        self.network.mlp_mean.my_layers[2].trainable_weights[0].assign(params_dict['network.mlp_mean.layers.2.weight'].T)  # kernel
        if OUTPUT_VARIABLES:
            print("before self.network.mlp_mean.my_layers[2].trainable_weights[1].assign(params_dict['network.mlp_mean.layers.2.bias'])")
        self.network.mlp_mean.my_layers[2].trainable_weights[1].assign(params_dict['network.mlp_mean.layers.2.bias'])     # bias







        if 'network.mlp_mean.layers.1.norm1.weight' in params_dict:
            self.network.mlp_mean.my_layers[1].norm1.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.1.norm1.weight'].T)  # kernel

        if 'network.mlp_mean.layers.1.norm1.bias' in params_dict:
            self.network.mlp_mean.my_layers[1].norm1.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.1.norm1.bias'])  # kernel


        if 'network.mlp_mean.layers.1.norm2.weight' in params_dict:
            self.network.mlp_mean.my_layers[1].norm2.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.1.norm2.weight'].T)  # kernel

        if 'network.mlp_mean.layers.1.norm2.bias' in params_dict:
            self.network.mlp_mean.my_layers[1].norm2.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.1.norm2.bias'])  # kernel





        if 'network.mlp_mean.layers.2.l1.weight' in params_dict:
            self.network.mlp_mean.my_layers[2].l1.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.2.l1.weight'].T)     # weight

        if 'network.mlp_mean.layers.2.l1.bias' in params_dict:
            self.network.mlp_mean.my_layers[2].l1.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.2.l1.bias'])     # bias

        if 'network.mlp_mean.layers.2.l2.weight' in params_dict:
            self.network.mlp_mean.my_layers[2].l2.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.2.l2.weight'].T)     # weight

        if 'network.mlp_mean.layers.2.l2.bias' in params_dict:
            self.network.mlp_mean.my_layers[2].l2.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.2.l2.bias'])     # bias

        if 'network.mlp_mean.layers.2.norm1.weight' in params_dict:
            self.network.mlp_mean.my_layers[2].norm1.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.2.norm1.weight'].T)     # weight

        if 'network.mlp_mean.layers.2.norm1.bias' in params_dict:
            self.network.mlp_mean.my_layers[2].norm1.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.2.norm1.bias'])     # bias

        if 'network.mlp_mean.layers.2.norm2.weight' in params_dict:
            self.network.mlp_mean.my_layers[2].norm2.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.2.norm2.weight'].T)     # weight

        if 'network.mlp_mean.layers.2.norm2.bias' in params_dict:
            self.network.mlp_mean.my_layers[2].norm2.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.2.norm2.bias'])     # bias



        if 'network.mlp_mean.layers.3.l1.weight' in params_dict:
            self.network.mlp_mean.my_layers[3].l1.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.3.l1.weight'].T)     # weight

        if 'network.mlp_mean.layers.3.l1.bias' in params_dict:
            self.network.mlp_mean.my_layers[3].l1.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.3.l1.bias'])     # bias

        if 'network.mlp_mean.layers.3.l2.weight' in params_dict:
            self.network.mlp_mean.my_layers[3].l2.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.3.l2.weight'].T)     # weight

        if 'network.mlp_mean.layers.3.l2.bias' in params_dict:
            self.network.mlp_mean.my_layers[3].l2.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.3.l2.bias'])     # bias

        if 'network.mlp_mean.layers.3.norm1.weight' in params_dict:
            self.network.mlp_mean.my_layers[3].norm1.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.3.norm1.weight'].T)     # weight

        if 'network.mlp_mean.layers.3.norm1.bias' in params_dict:
            self.network.mlp_mean.my_layers[3].norm1.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.3.norm1.bias'])     # bias

        if 'network.mlp_mean.layers.3.norm2.weight' in params_dict:
            self.network.mlp_mean.my_layers[3].norm2.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.3.norm2.weight'].T)     # weight

        if 'network.mlp_mean.layers.3.norm2.bias' in params_dict:
            self.network.mlp_mean.my_layers[3].norm2.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.3.norm2.bias'])     # bias


        if 'network.mlp_mean.layers.4.weight' in params_dict:
            self.network.mlp_mean.my_layers[4].trainable_weights[0].assign(params_dict['network.mlp_mean.layers.4.weight'].T)  # kernel

        if 'network.mlp_mean.layers.4.bias' in params_dict:
            self.network.mlp_mean.my_layers[4].trainable_weights[1].assign(params_dict['network.mlp_mean.layers.4.bias'])     # bias




    





    def load_pickle_diffusion_unet(self, network_path):
        pkl_file_path = network_path.replace('.pt', '_ema.pkl')

        print("pkl_file_path = ", pkl_file_path)

        import pickle
        with open(pkl_file_path, 'rb') as file:
            params_dict = pickle.load(file)

        if OUTPUT_VARIABLES:
            print("params_dict = ", params_dict)



        if 'network.time_mlp.1.weight' in params_dict:
            self.network.time_mlp[1].trainable_weights[0].assign(params_dict['network.time_mlp.1.weight'].T)  # kernel
        if 'network.time_mlp.1.bias' in params_dict:
            self.network.time_mlp[1].trainable_weights[1].assign(params_dict['network.time_mlp.1.bias'])  # bias
        if 'network.time_mlp.3.weight' in params_dict:
            self.network.time_mlp[3].trainable_weights[0].assign(params_dict['network.time_mlp.3.weight'].T)  # kernel
        if 'network.time_mlp.3.bias' in params_dict:
            self.network.time_mlp[3].trainable_weights[1].assign(params_dict['network.time_mlp.3.bias'])  # bias


        from util.torch_to_tf import nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d
        if 'network.mid_modules.0.blocks.0.block.0.weight' in params_dict:
            if isinstance(self.network.mid_modules[0].blocks[0].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[0].blocks[0].block[0].assign_torch_weights(params_dict['network.mid_modules.0.blocks.0.block.0.weight'])
            else:
                self.network.mid_modules[0].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.mid_modules.0.blocks.0.block.0.weight'])  # kernel
        if 'network.mid_modules.0.blocks.0.block.0.bias' in params_dict:
            self.network.mid_modules[0].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.mid_modules.0.blocks.0.block.0.bias'])  # bias

        if 'network.mid_modules.0.blocks.0.block.2.weight' in params_dict:
            if isinstance(self.network.mid_modules[0].blocks[0].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[0].blocks[0].block[2].assign_torch_weights(params_dict['network.mid_modules.0.blocks.0.block.2.weight'])
            else:
                self.network.mid_modules[0].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.mid_modules.0.blocks.0.block.2.weight'].T)  # kernel
        if 'network.mid_modules.0.blocks.0.block.2.bias' in params_dict:
            self.network.mid_modules[0].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.mid_modules.0.blocks.0.block.2.bias'])  # bias

        if 'network.mid_modules.0.blocks.1.block.0.weight' in params_dict:
            if isinstance(self.network.mid_modules[0].blocks[1].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[0].blocks[1].block[0].assign_torch_weights(params_dict['network.mid_modules.0.blocks.1.block.0.weight'])
            else:
                self.network.mid_modules[0].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.mid_modules.0.blocks.1.block.0.weight'].T)  # kernel
        if 'network.mid_modules.0.blocks.1.block.0.bias' in params_dict:
            self.network.mid_modules[0].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.mid_modules.0.blocks.1.block.0.bias'])  # bias
        
        if 'network.mid_modules.0.blocks.1.block.2.weight' in params_dict:
            if isinstance(self.network.mid_modules[0].blocks[1].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[0].blocks[1].block[2].assign_torch_weights(params_dict['network.mid_modules.0.blocks.1.block.2.weight'])
            else:
                self.network.mid_modules[0].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.mid_modules.0.blocks.1.block.2.weight'].T)  # kernel
        if 'network.mid_modules.0.blocks.1.block.2.bias' in params_dict:
            self.network.mid_modules[0].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.mid_modules.0.blocks.1.block.2.bias'])  # bias

        if 'network.mid_modules.0.cond_encoder.0.weight' in params_dict:
            if isinstance(self.network.mid_modules[0].cond_encoder[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[0].cond_encoder[0].assign_torch_weights(params_dict['network.mid_modules.0.cond_encoder.0.weight'])
            else:
                self.network.mid_modules[0].cond_encoder[0].trainable_weights[0].assign(params_dict['network.mid_modules.0.cond_encoder.0.weight'].T)  # kernel
        if 'network.mid_modules.0.cond_encoder.0.bias' in params_dict:
            self.network.mid_modules[0].cond_encoder[0].trainable_weights[1].assign(params_dict['network.mid_modules.0.cond_encoder.0.bias'])  # bias

        if 'network.mid_modules.0.cond_encoder.2.weight' in params_dict:
            if isinstance(self.network.mid_modules[0].cond_encoder[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[0].cond_encoder[2].assign_torch_weights(params_dict['network.mid_modules.0.cond_encoder.2.weight'])
            else:
                self.network.mid_modules[0].cond_encoder[2].trainable_weights[0].assign(params_dict['network.mid_modules.0.cond_encoder.2.weight'].T)  # kernel
        if 'network.mid_modules.0.cond_encoder.2.bias' in params_dict:
            self.network.mid_modules[0].cond_encoder[2].trainable_weights[1].assign(params_dict['network.mid_modules.0.cond_encoder.2.bias'])  # bias
        
        if 'network.mid_modules.0.cond_encoder.4.weight' in params_dict:
            if isinstance(self.network.mid_modules[0].cond_encoder[4], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[0].cond_encoder[4].assign_torch_weights(params_dict['network.mid_modules.0.cond_encoder.4.weight'])
            else:
                self.network.mid_modules[0].cond_encoder[4].trainable_weights[0].assign(params_dict['network.mid_modules.0.cond_encoder.4.weight'].T)  # kernel
        if 'network.mid_modules.0.cond_encoder.4.bias' in params_dict:
            self.network.mid_modules[0].cond_encoder[4].trainable_weights[1].assign(params_dict['network.mid_modules.0.cond_encoder.4.bias'])  # bias

        
        



        if 'network.mid_modules.1.blocks.0.block.0.weight' in params_dict:
            if isinstance(self.network.mid_modules[1].blocks[0].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[1].blocks[0].block[0].assign_torch_weights(params_dict['network.mid_modules.1.blocks.0.block.0.weight'])
            else:
                self.network.mid_modules[1].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.mid_modules.1.blocks.0.block.0.weight'].T)  # kernel
        if 'network.mid_modules.1.blocks.0.block.0.bias' in params_dict:
            self.network.mid_modules[1].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.mid_modules.1.blocks.0.block.0.bias'])  # bias

        if 'network.mid_modules.1.blocks.0.block.2.weight' in params_dict:
            if isinstance(self.network.mid_modules[1].blocks[0].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[1].blocks[0].block[2].assign_torch_weights(params_dict['network.mid_modules.1.blocks.0.block.2.weight'])
            else:
                self.network.mid_modules[1].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.mid_modules.1.blocks.0.block.2.weight'].T)  # kernel
        if 'network.mid_modules.1.blocks.0.block.2.bias' in params_dict:
            self.network.mid_modules[1].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.mid_modules.1.blocks.0.block.2.bias'])  # bias
        
        if 'network.mid_modules.1.blocks.1.block.0.weight' in params_dict:
            if isinstance(self.network.mid_modules[1].blocks[1].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[1].blocks[1].block[0].assign_torch_weights(params_dict['network.mid_modules.1.blocks.1.block.0.weight'])
            else:
                self.network.mid_modules[1].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.mid_modules.1.blocks.1.block.0.weight'].T)  # kernel
        if 'network.mid_modules.1.blocks.1.block.0.bias' in params_dict:
            self.network.mid_modules[1].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.mid_modules.1.blocks.1.block.0.bias'])  # bias
        
        if 'network.mid_modules.1.blocks.1.block.2.weight' in params_dict:
            if isinstance(self.network.mid_modules[1].blocks[1].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[1].blocks[1].block[2].assign_torch_weights(params_dict['network.mid_modules.1.blocks.1.block.2.weight'])
            else:
                self.network.mid_modules[1].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.mid_modules.1.blocks.1.block.2.weight'].T)  # kernel
        if 'network.mid_modules.1.blocks.1.block.2.bias' in params_dict:
            self.network.mid_modules[1].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.mid_modules.1.blocks.1.block.2.bias'])  # bias

        if 'network.mid_modules.1.cond_encoder.0.weight' in params_dict:
            if isinstance(self.network.mid_modules[1].cond_encoder[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[1].cond_encoder[0].assign_torch_weights(params_dict['network.mid_modules.1.cond_encoder.0.weight'])
            else:
                self.network.mid_modules[1].cond_encoder[0].trainable_weights[0].assign(params_dict['network.mid_modules.1.cond_encoder.0.weight'].T)  # kernel
        if 'network.mid_modules.1.cond_encoder.0.bias' in params_dict:
            self.network.mid_modules[1].cond_encoder[0].trainable_weights[1].assign(params_dict['network.mid_modules.1.cond_encoder.0.bias'])  # bias
        if 'network.mid_modules.1.cond_encoder.2.weight' in params_dict:

            if isinstance(self.network.mid_modules[1].cond_encoder[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[1].cond_encoder[2].assign_torch_weights(params_dict['network.mid_modules.1.cond_encoder.2.weight'])
            else:
                self.network.mid_modules[1].cond_encoder[2].trainable_weights[0].assign(params_dict['network.mid_modules.1.cond_encoder.2.weight'].T)  # kernel
        if 'network.mid_modules.1.cond_encoder.2.bias' in params_dict:
            self.network.mid_modules[1].cond_encoder[2].trainable_weights[1].assign(params_dict['network.mid_modules.1.cond_encoder.2.bias'])  # bias
        
        if 'network.mid_modules.1.cond_encoder.4.weight' in params_dict:
            if isinstance(self.network.mid_modules[1].cond_encoder[4], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.mid_modules[1].cond_encoder[4].assign_torch_weights(params_dict['network.mid_modules.1.cond_encoder.4.weight'])
            else:
                self.network.mid_modules[1].cond_encoder[4].trainable_weights[0].assign(params_dict['network.mid_modules.1.cond_encoder.4.weight'].T)  # kernel
        if 'network.mid_modules.1.cond_encoder.4.bias' in params_dict:
            self.network.mid_modules[1].cond_encoder[4].trainable_weights[1].assign(params_dict['network.mid_modules.1.cond_encoder.4.bias'])  # bias

        
        


        if 'network.down_modules.0.0.blocks.0.block.0.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[0].blocks[0].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[0].blocks[0].block[0].assign_torch_weights(params_dict['network.down_modules.0.0.blocks.0.block.0.weight'])
            else:
                self.network.down_modules[0].layers[0].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.down_modules.0.0.blocks.0.block.0.weight'].T)  # kernel
        
        if 'network.down_modules.0.0.blocks.0.block.0.bias' in params_dict:
            self.network.down_modules[0].layers[0].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.down_modules.0.0.blocks.0.block.0.bias'])  # bias
        
        if 'network.down_modules.0.0.blocks.0.block.2.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[0].blocks[0].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[0].blocks[0].block[2].assign_torch_weights(params_dict['network.down_modules.0.0.blocks.0.block.2.weight'])
            else:
                self.network.down_modules[0].layers[0].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.down_modules.0.0.blocks.0.block.2.weight'].T)  # kernel
        if 'network.down_modules.0.0.blocks.0.block.2.bias' in params_dict:
            self.network.down_modules[0].layers[0].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.down_modules.0.0.blocks.0.block.2.bias'])  # bias
        
        if 'network.down_modules.0.0.blocks.1.block.0.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[0].blocks[1].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[0].blocks[1].block[0].assign_torch_weights(params_dict['network.down_modules.0.0.blocks.1.block.0.weight'])
            else:
                self.network.down_modules[0].layers[0].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.down_modules.0.0.blocks.1.block.0.weight'].T)  # kernel
        if 'network.down_modules.0.0.blocks.1.block.0.bias' in params_dict:
            self.network.down_modules[0].layers[0].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.down_modules.0.0.blocks.1.block.0.bias'])  # bias
        
        
        if 'network.down_modules.0.0.blocks.1.block.2.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[0].blocks[1].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[0].blocks[1].block[2].assign_torch_weights(params_dict['network.down_modules.0.0.blocks.1.block.2.weight'])
            else:
                self.network.down_modules[0].layers[0].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.down_modules.0.0.blocks.1.block.2.weight'].T)  # kernel
        if 'network.down_modules.0.0.blocks.1.block.2.bias' in params_dict:
            self.network.down_modules[0].layers[0].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.down_modules.0.0.blocks.1.block.2.bias'])  # bias


        if 'network.down_modules.0.0.cond_encoder.0.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[0].cond_encoder[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[0].cond_encoder[0].assign_torch_weights(params_dict['network.down_modules.0.0.cond_encoder.0.weight'])
            else:
                self.network.down_modules[0].layers[0].cond_encoder[0].trainable_weights[0].assign(params_dict['network.down_modules.0.0.cond_encoder.0.weight'].T)  # kernel
        if 'network.down_modules.0.0.cond_encoder.0.bias' in params_dict:
            self.network.down_modules[0].layers[0].cond_encoder[0].trainable_weights[1].assign(params_dict['network.down_modules.0.0.cond_encoder.0.bias'])  # bias
        
        if 'network.down_modules.0.0.cond_encoder.2.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[0].cond_encoder[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[0].cond_encoder[2].assign_torch_weights(params_dict['network.down_modules.0.0.cond_encoder.2.weight'])
            else:
                self.network.down_modules[0].layers[0].cond_encoder[2].trainable_weights[0].assign(params_dict['network.down_modules.0.0.cond_encoder.2.weight'].T)  # kernel
        if 'network.down_modules.0.0.cond_encoder.2.bias' in params_dict:
            self.network.down_modules[0].layers[0].cond_encoder[2].trainable_weights[1].assign(params_dict['network.down_modules.0.0.cond_encoder.2.bias'])  # bias
        
        if 'network.down_modules.0.0.cond_encoder.4.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[0].cond_encoder[4], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[0].cond_encoder[4].assign_torch_weights(params_dict['network.down_modules.0.0.cond_encoder.4.weight'])
            else:
                self.network.down_modules[0].layers[0].cond_encoder[4].trainable_weights[0].assign(params_dict['network.down_modules.0.0.cond_encoder.4.weight'].T)  # kernel
        if 'network.down_modules.0.0.cond_encoder.4.bias' in params_dict:
            self.network.down_modules[0].layers[0].cond_encoder[4].trainable_weights[1].assign(params_dict['network.down_modules.0.0.cond_encoder.4.bias'])  # bias


        if 'network.down_modules.0.0.residual_conv.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[0].residual_conv, (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[0].residual_conv.assign_torch_weights(params_dict['network.down_modules.0.0.residual_conv.weight'])
            else:
                self.network.down_modules[0].layers[0].residual_conv.trainable_weights[0].assign(params_dict['network.down_modules.0.0.residual_conv.weight'].T)  # kernel
        if 'network.down_modules.0.0.residual_conv.bias' in params_dict:
            self.network.down_modules[0].layers[0].residual_conv.trainable_weights[1].assign(params_dict['network.down_modules.0.0.residual_conv.bias'])  # bias








        if 'network.down_modules.0.1.blocks.0.block.0.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[1].blocks[0].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[1].blocks[0].block[0].assign_torch_weights(params_dict['network.down_modules.0.1.blocks.0.block.0.weight'])
            else:
                self.network.down_modules[0].layers[1].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.down_modules.0.1.blocks.0.block.0.weight'].T)  # kernel
        if 'network.down_modules.0.1.blocks.0.block.0.bias' in params_dict:
            self.network.down_modules[0].layers[1].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.down_modules.0.1.blocks.0.block.0.bias'])  # bias
        
        if 'network.down_modules.0.1.blocks.0.block.2.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[1].blocks[0].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[1].blocks[0].block[2].assign_torch_weights(params_dict['network.down_modules.0.1.blocks.0.block.2.weight'])
            else:
                self.network.down_modules[0].layers[1].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.down_modules.0.1.blocks.0.block.2.weight'].T)  # kernel
        if 'network.down_modules.0.1.blocks.0.block.2.bias' in params_dict:
            self.network.down_modules[0].layers[1].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.down_modules.0.1.blocks.0.block.2.bias'])  # bias
        
        if 'network.down_modules.0.1.blocks.1.block.0.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[1].blocks[1].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[1].blocks[1].block[0].assign_torch_weights(params_dict['network.down_modules.0.1.blocks.1.block.0.weight'])
            else:
                self.network.down_modules[0].layers[1].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.down_modules.0.1.blocks.1.block.0.weight'].T)  # kernel
        if 'network.down_modules.0.1.blocks.1.block.0.bias' in params_dict:
            self.network.down_modules[0].layers[1].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.down_modules.0.1.blocks.1.block.0.bias'])  # bias
        
        if 'network.down_modules.0.1.blocks.1.block.2.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[1].blocks[1].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[1].blocks[1].block[2].assign_torch_weights(params_dict['network.down_modules.0.1.blocks.1.block.2.weight'])
            else:
                self.network.down_modules[0].layers[1].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.down_modules.0.1.blocks.1.block.2.weight'].T)  # kernel
        if 'network.down_modules.0.1.blocks.1.block.2.bias' in params_dict:
            self.network.down_modules[0].layers[1].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.down_modules.0.1.blocks.1.block.2.bias'])  # bias


        if 'network.down_modules.0.1.cond_encoder.0.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[1].cond_encoder[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[1].cond_encoder[0].assign_torch_weights(params_dict['network.down_modules.0.1.cond_encoder.0.weight'])
            else:
                self.network.down_modules[0].layers[1].cond_encoder[0].trainable_weights[0].assign(params_dict['network.down_modules.0.1.cond_encoder.0.weight'].T)  # kernel
        if 'network.down_modules.0.1.cond_encoder.0.bias' in params_dict:
            self.network.down_modules[0].layers[1].cond_encoder[0].trainable_weights[1].assign(params_dict['network.down_modules.0.1.cond_encoder.0.bias'])  # bias
        
        if 'network.down_modules.0.1.cond_encoder.2.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[1].cond_encoder[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[1].cond_encoder[2].assign_torch_weights(params_dict['network.down_modules.0.1.cond_encoder.2.weight'])
            else:
                self.network.down_modules[0].layers[1].cond_encoder[2].trainable_weights[0].assign(params_dict['network.down_modules.0.1.cond_encoder.2.weight'].T)  # kernel
        if 'network.down_modules.0.1.cond_encoder.2.bias' in params_dict:
            self.network.down_modules[0].layers[1].cond_encoder[2].trainable_weights[1].assign(params_dict['network.down_modules.0.1.cond_encoder.2.bias'])  # bias
        
        if 'network.down_modules.0.1.cond_encoder.4.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[1].cond_encoder[4], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[1].cond_encoder[4].assign_torch_weights(params_dict['network.down_modules.0.1.cond_encoder.4.weight'])
            else:
                self.network.down_modules[0].layers[1].cond_encoder[4].trainable_weights[0].assign(params_dict['network.down_modules.0.1.cond_encoder.4.weight'].T)  # kernel
        if 'network.down_modules.0.1.cond_encoder.4.bias' in params_dict:
            self.network.down_modules[0].layers[1].cond_encoder[4].trainable_weights[1].assign(params_dict['network.down_modules.0.1.cond_encoder.4.bias'])  # bias


        if 'network.down_modules.0.2.conv.weight' in params_dict:
            if isinstance(self.network.down_modules[0].layers[2].conv, (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[0].layers[2].conv.assign_torch_weights(params_dict['network.down_modules.0.2.conv.weight'])
            else:
                self.network.down_modules[0].layers[2].conv.trainable_weights[0].assign(params_dict['network.down_modules.0.2.conv.weight'].T)  # kernel
        if 'network.down_modules.0.2.conv.bias' in params_dict:
            self.network.down_modules[0].layers[2].conv.trainable_weights[1].assign(params_dict['network.down_modules.0.2.conv.bias'])  # bias












        if 'network.down_modules.1.0.blocks.0.block.0.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[0].blocks[0].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[0].blocks[0].block[0].assign_torch_weights(params_dict['network.down_modules.1.0.blocks.0.block.0.weight'])
            else:
                self.network.down_modules[1].layers[0].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.down_modules.1.0.blocks.0.block.0.weight'].T)  # kernel
        if 'network.down_modules.1.0.blocks.0.block.0.bias' in params_dict:
            self.network.down_modules[1].layers[0].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.down_modules.1.0.blocks.0.block.0.bias'])  # bias
        
        if 'network.down_modules.1.0.blocks.0.block.2.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[0].blocks[0].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[0].blocks[0].block[2].assign_torch_weights(params_dict['network.down_modules.1.0.blocks.0.block.2.weight'])
            else:
                self.network.down_modules[1].layers[0].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.down_modules.1.0.blocks.0.block.2.weight'].T)  # kernel
        if 'network.down_modules.1.0.blocks.0.block.2.bias' in params_dict:
            self.network.down_modules[1].layers[0].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.down_modules.1.0.blocks.0.block.2.bias'])  # bias
        
        if 'network.down_modules.1.0.blocks.1.block.0.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[0].blocks[1].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[0].blocks[1].block[0].assign_torch_weights(params_dict['network.down_modules.1.0.blocks.1.block.0.weight'])
            else:
                self.network.down_modules[1].layers[0].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.down_modules.1.0.blocks.1.block.0.weight'].T)  # kernel
        if 'network.down_modules.1.0.blocks.1.block.0.bias' in params_dict:
            self.network.down_modules[1].layers[0].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.down_modules.1.0.blocks.1.block.0.bias'])  # bias
        
        if 'network.down_modules.1.0.blocks.1.block.2.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[0].blocks[1].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[0].blocks[1].block[2].assign_torch_weights(params_dict['network.down_modules.1.0.blocks.1.block.2.weight'])
            else:
                self.network.down_modules[1].layers[0].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.down_modules.1.0.blocks.1.block.2.weight'].T)  # kernel
        if 'network.down_modules.1.0.blocks.1.block.2.bias' in params_dict:
            self.network.down_modules[1].layers[0].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.down_modules.1.0.blocks.1.block.2.bias'])  # bias

        if 'network.down_modules.1.0.cond_encoder.0.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[0].cond_encoder[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[0].cond_encoder[0].assign_torch_weights(params_dict['network.down_modules.1.0.cond_encoder.0.weight'])
            else:
                self.network.down_modules[1].layers[0].cond_encoder[0].trainable_weights[0].assign(params_dict['network.down_modules.1.0.cond_encoder.0.weight'].T)  # kernel
        if 'network.down_modules.1.0.cond_encoder.0.bias' in params_dict:
            self.network.down_modules[1].layers[0].cond_encoder[0].trainable_weights[1].assign(params_dict['network.down_modules.1.0.cond_encoder.0.bias'])  # bias
        
        if 'network.down_modules.1.0.cond_encoder.2.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[0].cond_encoder[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[0].cond_encoder[2].assign_torch_weights(params_dict['network.down_modules.1.0.cond_encoder.2.weight'])
            else:
                self.network.down_modules[1].layers[0].cond_encoder[2].trainable_weights[0].assign(params_dict['network.down_modules.1.0.cond_encoder.2.weight'].T)  # kernel
        if 'network.down_modules.1.0.cond_encoder.2.bias' in params_dict:
            self.network.down_modules[1].layers[0].cond_encoder[2].trainable_weights[1].assign(params_dict['network.down_modules.1.0.cond_encoder.2.bias'])  # bias
        
        if 'network.down_modules.1.0.cond_encoder.4.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[0].cond_encoder[4], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[0].cond_encoder[4].assign_torch_weights(params_dict['network.down_modules.1.0.cond_encoder.4.weight'])
            else:
                self.network.down_modules[1].layers[0].cond_encoder[4].trainable_weights[0].assign(params_dict['network.down_modules.1.0.cond_encoder.4.weight'].T)  # kernel
        if 'network.down_modules.1.0.cond_encoder.4.bias' in params_dict:
            self.network.down_modules[1].layers[0].cond_encoder[4].trainable_weights[1].assign(params_dict['network.down_modules.1.0.cond_encoder.4.bias'])  # bias


        if 'network.down_modules.1.0.residual_conv.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[0].residual_conv, (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[0].residual_conv.assign_torch_weights(params_dict['network.down_modules.1.0.residual_conv.weight'])
            else:
                self.network.down_modules[1].layers[0].residual_conv.trainable_weights[0].assign(params_dict['network.down_modules.1.0.residual_conv.weight'].T)  # kernel
        if 'network.down_modules.1.0.residual_conv.bias' in params_dict:
            self.network.down_modules[1].layers[0].residual_conv.trainable_weights[1].assign(params_dict['network.down_modules.1.0.residual_conv.bias'])  # bias





        if 'network.down_modules.1.1.blocks.0.block.0.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[1].blocks[0].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[1].blocks[0].block[0].assign_torch_weights(params_dict['network.down_modules.1.1.blocks.0.block.0.weight'])
            else:
                self.network.down_modules[1].layers[1].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.down_modules.1.1.blocks.0.block.0.weight'].T)  # kernel
        if 'network.down_modules.1.1.blocks.0.block.0.bias' in params_dict:
            self.network.down_modules[1].layers[1].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.down_modules.1.1.blocks.0.block.0.bias'])  # bias
        
        if 'network.down_modules.1.1.blocks.0.block.2.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[1].blocks[0].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[1].blocks[0].block[2].assign_torch_weights(params_dict['network.down_modules.1.1.blocks.0.block.2.weight'])
            else:
                self.network.down_modules[1].layers[1].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.down_modules.1.1.blocks.0.block.2.weight'].T)  # kernel
        if 'network.down_modules.1.1.blocks.0.block.2.bias' in params_dict:
            self.network.down_modules[1].layers[1].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.down_modules.1.1.blocks.0.block.2.bias'])  # bias
        
        if 'network.down_modules.1.1.blocks.1.block.0.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[1].blocks[1].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[1].blocks[1].block[0].assign_torch_weights(params_dict['network.down_modules.1.1.blocks.1.block.0.weight'])
            else:
                self.network.down_modules[1].layers[1].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.down_modules.1.1.blocks.1.block.0.weight'].T)  # kernel
        if 'network.down_modules.1.1.blocks.1.block.0.bias' in params_dict:
            self.network.down_modules[1].layers[1].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.down_modules.1.1.blocks.1.block.0.bias'])  # bias
        
        if 'network.down_modules.1.1.blocks.1.block.2.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[1].blocks[1].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[1].blocks[1].block[2].assign_torch_weights(params_dict['network.down_modules.1.1.blocks.1.block.2.weight'])
            else:
                self.network.down_modules[1].layers[1].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.down_modules.1.1.blocks.1.block.2.weight'].T)  # kernel
        if 'network.down_modules.1.1.blocks.1.block.2.bias' in params_dict:
            self.network.down_modules[1].layers[1].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.down_modules.1.1.blocks.1.block.2.bias'])  # bias


        if 'network.down_modules.1.1.cond_encoder.0.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[1].cond_encoder[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[1].cond_encoder[0].assign_torch_weights(params_dict['network.down_modules.1.1.cond_encoder.0.weight'])
            else:
                self.network.down_modules[1].layers[1].cond_encoder[0].trainable_weights[0].assign(params_dict['network.down_modules.1.1.cond_encoder.0.weight'].T)  # kernel
        if 'network.down_modules.1.1.cond_encoder.0.bias' in params_dict:
            self.network.down_modules[1].layers[1].cond_encoder[0].trainable_weights[1].assign(params_dict['network.down_modules.1.1.cond_encoder.0.bias'])  # bias
        
        if 'network.down_modules.1.1.cond_encoder.2.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[1].cond_encoder[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[1].cond_encoder[2].assign_torch_weights(params_dict['network.down_modules.1.1.cond_encoder.2.weight'])
            else:
                self.network.down_modules[1].layers[1].cond_encoder[2].trainable_weights[0].assign(params_dict['network.down_modules.1.1.cond_encoder.2.weight'].T)  # kernel
        if 'network.down_modules.1.1.cond_encoder.2.bias' in params_dict:
            self.network.down_modules[1].layers[1].cond_encoder[2].trainable_weights[1].assign(params_dict['network.down_modules.1.1.cond_encoder.2.bias'])  # bias
        
        if 'network.down_modules.1.1.cond_encoder.4.weight' in params_dict:
            if isinstance(self.network.down_modules[1].layers[1].cond_encoder[4], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.down_modules[1].layers[1].cond_encoder[4].assign_torch_weights(params_dict['network.down_modules.1.1.cond_encoder.4.weight'])
            else:
                self.network.down_modules[1].layers[1].cond_encoder[4].trainable_weights[0].assign(params_dict['network.down_modules.1.1.cond_encoder.4.weight'].T)  # kernel
        if 'network.down_modules.1.1.cond_encoder.4.bias' in params_dict:
            self.network.down_modules[1].layers[1].cond_encoder[4].trainable_weights[1].assign(params_dict['network.down_modules.1.1.cond_encoder.4.bias'])  # bias










        if 'network.up_modules.0.0.blocks.0.block.0.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[0].blocks[0].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[0].blocks[0].block[0].assign_torch_weights(params_dict['network.up_modules.0.0.blocks.0.block.0.weight'])
            else:
                self.network.up_modules[0].layers[0].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.up_modules.0.0.blocks.0.block.0.weight'].T)  # kernel
        if 'network.up_modules.0.0.blocks.0.block.0.bias' in params_dict:
            self.network.up_modules[0].layers[0].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.up_modules.0.0.blocks.0.block.0.bias'])  # bias
        
        if 'network.up_modules.0.0.blocks.0.block.2.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[0].blocks[0].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[0].blocks[0].block[2].assign_torch_weights(params_dict['network.up_modules.0.0.blocks.0.block.2.weight'])
            else:
                self.network.up_modules[0].layers[0].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.up_modules.0.0.blocks.0.block.2.weight'].T)  # kernel
        if 'network.up_modules.0.0.blocks.0.block.2.bias' in params_dict:
            self.network.up_modules[0].layers[0].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.up_modules.0.0.blocks.0.block.2.bias'])  # bias
        
        
        if 'network.up_modules.0.0.blocks.1.block.0.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[0].blocks[1].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[0].blocks[1].block[0].assign_torch_weights(params_dict['network.up_modules.0.0.blocks.1.block.0.weight'])
            else:
                self.network.up_modules[0].layers[0].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.up_modules.0.0.blocks.1.block.0.weight'].T)  # kernel
        if 'network.up_modules.0.0.blocks.1.block.0.bias' in params_dict:
            self.network.up_modules[0].layers[0].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.up_modules.0.0.blocks.1.block.0.bias'])  # bias
        
        
        if 'network.up_modules.0.0.blocks.1.block.2.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[0].blocks[1].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[0].blocks[1].block[2].assign_torch_weights(params_dict['network.up_modules.0.0.blocks.1.block.2.weight'])
            else:
                self.network.up_modules[0].layers[0].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.up_modules.0.0.blocks.1.block.2.weight'].T)  # kernel
        if 'network.up_modules.0.0.blocks.1.block.2.bias' in params_dict:
            self.network.up_modules[0].layers[0].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.up_modules.0.0.blocks.1.block.2.bias'])  # bias

        if 'network.up_modules.0.0.cond_encoder.0.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[0].cond_encoder[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[0].cond_encoder[0].assign_torch_weights(params_dict['network.up_modules.0.0.cond_encoder.0.weight'])
            else:
                self.network.up_modules[0].layers[0].cond_encoder[0].trainable_weights[0].assign(params_dict['network.up_modules.0.0.cond_encoder.0.weight'].T)  # kernel
        if 'network.up_modules.0.0.cond_encoder.0.bias' in params_dict:
            self.network.up_modules[0].layers[0].cond_encoder[0].trainable_weights[1].assign(params_dict['network.up_modules.0.0.cond_encoder.0.bias'])  # bias
        
        if 'network.up_modules.0.0.cond_encoder.2.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[0].cond_encoder[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[0].cond_encoder[2].assign_torch_weights(params_dict['network.up_modules.0.0.cond_encoder.2.weight'])
            else:
                self.network.up_modules[0].layers[0].cond_encoder[2].trainable_weights[0].assign(params_dict['network.up_modules.0.0.cond_encoder.2.weight'].T)  # kernel
        if 'network.up_modules.0.0.cond_encoder.2.bias' in params_dict:
            self.network.up_modules[0].layers[0].cond_encoder[2].trainable_weights[1].assign(params_dict['network.up_modules.0.0.cond_encoder.2.bias'])  # bias
        
        
        if 'network.up_modules.0.0.cond_encoder.4.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[0].cond_encoder[4], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[0].cond_encoder[4].assign_torch_weights(params_dict['network.up_modules.0.0.cond_encoder.4.weight'])
            else:
                self.network.up_modules[0].layers[0].cond_encoder[4].trainable_weights[0].assign(params_dict['network.up_modules.0.0.cond_encoder.4.weight'].T)  # kernel
        if 'network.up_modules.0.0.cond_encoder.4.bias' in params_dict:
            self.network.up_modules[0].layers[0].cond_encoder[4].trainable_weights[1].assign(params_dict['network.up_modules.0.0.cond_encoder.4.bias'])  # bias


        if 'network.up_modules.0.0.residual_conv.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[0].residual_conv, (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[0].residual_conv.assign_torch_weights(params_dict['network.up_modules.0.0.residual_conv.weight'])
            else:
                self.network.up_modules[0].layers[0].residual_conv.trainable_weights[0].assign(params_dict['network.up_modules.0.0.residual_conv.weight'].T)  # kernel
        if 'network.up_modules.0.0.residual_conv.bias' in params_dict:
            self.network.up_modules[0].layers[0].residual_conv.trainable_weights[1].assign(params_dict['network.up_modules.0.0.residual_conv.bias'])  # bias



        if 'network.up_modules.0.1.blocks.0.block.0.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[1].blocks[0].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[1].blocks[0].block[0].assign_torch_weights(params_dict['network.up_modules.0.1.blocks.0.block.0.weight'])
            else:
                self.network.up_modules[0].layers[1].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.up_modules.0.1.blocks.0.block.0.weight'].T)  # kernel
        if 'network.up_modules.0.1.blocks.0.block.0.bias' in params_dict:
            self.network.up_modules[0].layers[1].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.up_modules.0.1.blocks.0.block.0.bias'])  # bias
        
        if 'network.up_modules.0.1.blocks.0.block.2.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[1].blocks[0].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[1].blocks[0].block[2].assign_torch_weights(params_dict['network.up_modules.0.1.blocks.0.block.2.weight'])
            else:
                self.network.up_modules[0].layers[1].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.up_modules.0.1.blocks.0.block.2.weight'].T)  # kernel
        if 'network.up_modules.0.1.blocks.0.block.2.bias' in params_dict:
            self.network.up_modules[0].layers[1].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.up_modules.0.1.blocks.0.block.2.bias'])  # bias
        
        if 'network.up_modules.0.1.blocks.1.block.0.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[1].blocks[1].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[1].blocks[1].block[0].assign_torch_weights(params_dict['network.up_modules.0.1.blocks.1.block.0.weight'])
            else:
                self.network.up_modules[0].layers[1].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.up_modules.0.1.blocks.1.block.0.weight'].T)  # kernel
        if 'network.up_modules.0.1.blocks.1.block.0.bias' in params_dict:
            self.network.up_modules[0].layers[1].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.up_modules.0.1.blocks.1.block.0.bias'])  # bias
        
        
        if 'network.up_modules.0.1.blocks.1.block.2.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[1].blocks[1].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[1].blocks[1].block[2].assign_torch_weights(params_dict['network.up_modules.0.1.blocks.1.block.2.weight'])
            else:
                self.network.up_modules[0].layers[1].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.up_modules.0.1.blocks.1.block.2.weight'].T)  # kernel
        if 'network.up_modules.0.1.blocks.1.block.2.bias' in params_dict:
            self.network.up_modules[0].layers[1].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.up_modules.0.1.blocks.1.block.2.bias'])  # bias

        if 'network.up_modules.0.1.cond_encoder.0.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[1].cond_encoder[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[1].cond_encoder[0].assign_torch_weights(params_dict['network.up_modules.0.1.cond_encoder.0.weight'])
            else:
                self.network.up_modules[0].layers[1].cond_encoder[0].trainable_weights[0].assign(params_dict['network.up_modules.0.1.cond_encoder.0.weight'].T)  # kernel
        if 'network.up_modules.0.1.cond_encoder.0.bias' in params_dict:
            self.network.up_modules[0].layers[1].cond_encoder[0].trainable_weights[1].assign(params_dict['network.up_modules.0.1.cond_encoder.0.bias'])  # bias
        
        
        
        if 'network.up_modules.0.1.cond_encoder.2.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[1].cond_encoder[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[1].cond_encoder[2].assign_torch_weights(params_dict['network.up_modules.0.1.cond_encoder.2.weight'])
            else:
                self.network.up_modules[0].layers[1].cond_encoder[2].trainable_weights[0].assign(params_dict['network.up_modules.0.1.cond_encoder.2.weight'].T)  # kernel
        if 'network.up_modules.0.1.cond_encoder.2.bias' in params_dict:
            self.network.up_modules[0].layers[1].cond_encoder[2].trainable_weights[1].assign(params_dict['network.up_modules.0.1.cond_encoder.2.bias'])  # bias
        
        if 'network.up_modules.0.1.cond_encoder.4.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[1].cond_encoder[4], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[1].cond_encoder[4].assign_torch_weights(params_dict['network.up_modules.0.1.cond_encoder.4.weight'])
            else:
                self.network.up_modules[0].layers[1].cond_encoder[4].trainable_weights[0].assign(params_dict['network.up_modules.0.1.cond_encoder.4.weight'].T)  # kernel
        if 'network.up_modules.0.1.cond_encoder.4.bias' in params_dict:
            self.network.up_modules[0].layers[1].cond_encoder[4].trainable_weights[1].assign(params_dict['network.up_modules.0.1.cond_encoder.4.bias'])  # bias


        if 'network.up_modules.0.2.conv.weight' in params_dict:
            if isinstance(self.network.up_modules[0].layers[2].conv, (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.up_modules[0].layers[2].conv.assign_torch_weights(params_dict['network.up_modules.0.2.conv.weight'])
            else:
                self.network.up_modules[0].layers[2].conv.trainable_weights[0].assign(params_dict['network.up_modules.0.2.conv.weight'].T)  # kernel
        if 'network.up_modules.0.2.conv.bias' in params_dict:
            self.network.up_modules[0].layers[2].conv.trainable_weights[1].assign(params_dict['network.up_modules.0.2.conv.bias'])  # bias




        if 'network.final_conv.0.block.0.weight':
            if isinstance(self.network.final_conv[0].block[0], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.final_conv[0].block[0].assign_torch_weights(params_dict['network.final_conv.0.block.0.weight'])
            else:
                self.network.final_conv[0].block[0].trainable_weights[0].assign(params_dict['network.final_conv.0.block.0.weight'].T)  # kernel
        if 'network.final_conv.0.block.0.bias':
            self.network.final_conv[0].block[0].trainable_weights[1].assign(params_dict['network.final_conv.0.block.0.bias'])  # bias

        if 'network.final_conv.0.block.2.weight':
            if isinstance(self.network.final_conv[0].block[2], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.final_conv[0].block[2].assign_torch_weights(params_dict['network.final_conv.0.block.2.weight'])
            else:
                self.network.final_conv[0].block[2].trainable_weights[0].assign(params_dict['network.final_conv.0.block.2.weight'].T)  # kernel
        if 'network.final_conv.0.block.2.bias':
            self.network.final_conv[0].block[2].trainable_weights[1].assign(params_dict['network.final_conv.0.block.2.bias'])  # bias


        if 'network.final_conv.1.weight':
            if isinstance(self.network.final_conv[1], (nn_Conv1d, nn_Conv2d, nn_ConvTranspose1d)):
                self.network.final_conv[1].assign_torch_weights(params_dict['network.final_conv.1.weight'])
            else:
                self.network.final_conv[1].trainable_weights[0].assign(params_dict['network.final_conv.1.weight'].T)  # kernel
        if 'network.final_conv.1.bias':
            self.network.final_conv[1].trainable_weights[1].assign(params_dict['network.final_conv.1.bias'])  # bias












    def load_pickle_diffusion_unet_img(self, network_path):
        pkl_file_path = network_path.replace('.pt', '_ema.pkl')

        print("pkl_file_path = ", pkl_file_path)

        import pickle
        with open(pkl_file_path, 'rb') as file:
            params_dict = pickle.load(file)




        if OUTPUT_VARIABLES:
            print("params_dict = ", params_dict)

        
        




        if 'network.backbone.vit.pos_embed' in params_dict:
            self.network.backbone.vit.pos_embed = nn_Parameter( torch_tensor(params_dict['network.backbone.vit.pos_embed']) )
            
        if 'network.backbone.vit.patch_embed.embed.0.weight' in params_dict:
            self.network.backbone.vit.patch_embed.embed[0].trainable_weights[0].assign(params_dict['network.backbone.vit.patch_embed.embed.0.weight'].T)  # kernel
        if 'network.backbone.vit.patch_embed.embed.0.bias' in params_dict:
            self.network.backbone.vit.patch_embed.embed[0].trainable_weights[1].assign(params_dict['network.backbone.vit.patch_embed.embed.0.bias'])  # bias

        if 'network.backbone.vit.patch_embed.embed.3.weight' in params_dict:
            self.network.backbone.vit.patch_embed.embed[3].trainable_weights[0].assign(params_dict['network.backbone.vit.patch_embed.embed.3.weight'].T)  # kernel
        if 'network.backbone.vit.patch_embed.embed.3.bias' in params_dict:
            self.network.backbone.vit.patch_embed.embed[3].trainable_weights[1].assign(params_dict['network.backbone.vit.patch_embed.embed.3.bias'])  # bias




        if 'network.backbone.vit.net.0.layer_norm1.weight' in params_dict:
            self.network.backbone.vit.net[0].layer_norm1.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.layer_norm1.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.layer_norm1.bias' in params_dict:
            self.network.backbone.vit.net[0].layer_norm1.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.layer_norm1.bias'])  # bias

        if 'network.backbone.vit.net.0.mha.qkv_proj.weight' in params_dict:
            self.network.backbone.vit.net[0].mha.qkv_proj.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.mha.qkv_proj.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.mha.qkv_proj.bias' in params_dict:
            self.network.backbone.vit.net[0].mha.qkv_proj.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.mha.qkv_proj.bias'])  # bias


        if 'network.backbone.vit.net.0.mha.out_proj.weight' in params_dict:
            self.network.backbone.vit.net[0].mha.out_proj.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.mha.out_proj.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.mha.out_proj.bias' in params_dict:
            self.network.backbone.vit.net[0].mha.out_proj.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.mha.out_proj.bias'])  # bias

        if 'network.backbone.vit.net.0.layer_norm2.weight' in params_dict:
            self.network.backbone.vit.net[0].layer_norm2.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.layer_norm2.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.layer_norm2.bias' in params_dict:
            self.network.backbone.vit.net[0].layer_norm2.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.layer_norm2.bias'])  # bias


        if 'network.backbone.vit.net.0.linear1.weight' in params_dict:
            self.network.backbone.vit.net[0].linear1.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.linear1.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.linear1.bias' in params_dict:
            self.network.backbone.vit.net[0].linear1.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.linear1.bias'])  # bias


        if 'network.backbone.vit.net.0.linear2.weight' in params_dict:
            self.network.backbone.vit.net[0].linear2.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.linear2.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.linear2.bias' in params_dict:
            self.network.backbone.vit.net[0].linear2.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.linear2.bias'])  # bias


        if 'network.backbone.vit.norm.weight' in params_dict:
            self.network.backbone.vit.norm.trainable_weights[0].assign(params_dict['network.backbone.vit.norm.weight'].T)  # kernel
        if 'network.backbone.vit.norm.bias' in params_dict:
            self.network.backbone.vit.norm.trainable_weights[1].assign(params_dict['network.backbone.vit.norm.bias'])  # bias





        print("self.network.compress = ", self.network.compress)
        assert 0 == 1, "network.compress check"
        if 'network.compress.weight' in params_dict:
            self.network.compress.weight.trainable_weights[0].assign(params_dict['network.compress.weight'])  # kernel

        if 'network.compress.input_proj.0.weight' in params_dict:
            self.network.compress.input_proj[0].trainable_weights[0].assign(params_dict['network.compress.input_proj.0.weight'].T)  # kernel
        if 'network.compress.input_proj.0.bias' in params_dict:
            self.network.compress.input_proj[0].trainable_weights[1].assign(params_dict['network.compress.input_proj.0.bias'])  # bias

        if 'network.compress.input_proj.1.weight' in params_dict:
            self.network.compress.input_proj[1].trainable_weights[0].assign(params_dict['network.compress.input_proj.1.weight'].T)  # kernel
        if 'network.compress.input_proj.1.bias' in params_dict:
            self.network.compress.input_proj[1].trainable_weights[1].assign(params_dict['network.compress.input_proj.1.bias'])  # bias


        if 'network.compress1.weight' in params_dict:
            self.network.compress1.trainable_weights[0].assign(params_dict['network.compress1.weight'])  # kernel

        if 'network.compress1.input_proj.0.weight' in params_dict:
            self.network.compress1.input_proj[0].trainable_weights[0].assign(params_dict['network.compress1.input_proj.0.weight'].T)  # kernel
        if 'network.compress1.input_proj.0.bias' in params_dict:
            self.network.compress1.input_proj[0].trainable_weights[1].assign(params_dict['network.compress1.input_proj.0.bias'])  # bias

        if 'network.compress1.input_proj.1.weight' in params_dict:
            self.network.compress1.input_proj[1].trainable_weights[0].assign(params_dict['network.compress1.input_proj.1.weight'].T)  # kernel
        if 'network.compress1.input_proj.1.bias' in params_dict:
            self.network.compress1.input_proj[1].trainable_weights[1].assign(params_dict['network.compress1.input_proj.1.bias'])  # bias


        if 'network.compress2.weight' in params_dict:
            self.network.compress2.trainable_weights[0].assign(params_dict['network.compress2.weight'])  # kernel

        if 'network.compress2.input_proj.0.weight' in params_dict:
            self.network.compress2.input_proj[0].trainable_weights[0].assign(params_dict['network.compress2.input_proj.0.weight'].T)  # kernel
        if 'network.compress2.input_proj.0.bias' in params_dict:
            self.network.compress2.input_proj[0].trainable_weights[1].assign(params_dict['network.compress2.input_proj.0.bias'])  # bias

        if 'network.compress2.input_proj.1.weight' in params_dict:
            self.network.compress2.input_proj[1].trainable_weights[0].assign(params_dict['network.compress2.input_proj.1.weight'].T)  # kernel
        if 'network.compress2.input_proj.1.bias' in params_dict:
            self.network.compress2.input_proj[1].trainable_weights[1].assign(params_dict['network.compress2.input_proj.1.bias'])  # bias



        if 'network.time_mlp.1.weight' in params_dict:
            self.network.time_mlp[1].trainable_weights[0].assign(params_dict['network.time_mlp.1.weight'].T)  # kernel
        if 'network.time_mlp.1.bias' in params_dict:
            self.network.time_mlp[1].trainable_weights[1].assign(params_dict['network.time_mlp.1.bias'])  # bias

        if 'network.time_mlp.3.weight' in params_dict:
            self.network.time_mlp[3].trainable_weights[0].assign(params_dict['network.time_mlp.3.weight'].T)  # kernel
        if 'network.time_mlp.3.bias' in params_dict:
            self.network.time_mlp[3].trainable_weights[1].assign(params_dict['network.time_mlp.3.bias'])  # bias






        if 'network.mid_modules.0.blocks.0.block.0.weight' in params_dict:
            self.network.mid_modules[0].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.mid_modules.0.blocks.0.block.0.weight'].T)  # kernel
        if 'network.mid_modules.0.blocks.0.block.0.bias' in params_dict:
            self.network.mid_modules[0].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.mid_modules.0.blocks.0.block.0.bias'])  # bias
        if 'network.mid_modules.0.blocks.0.block.2.weight' in params_dict:
            self.network.mid_modules[0].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.mid_modules.0.blocks.0.block.2.weight'].T)  # kernel
        if 'network.mid_modules.0.blocks.0.block.2.bias' in params_dict:
            self.network.mid_modules[0].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.mid_modules.0.blocks.0.block.2.bias'])  # bias
        if 'network.mid_modules.0.blocks.1.block.0.weight' in params_dict:
            self.network.mid_modules[0].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.mid_modules.0.blocks.1.block.0.weight'].T)  # kernel
        if 'network.mid_modules.0.blocks.1.block.0.bias' in params_dict:
            self.network.mid_modules[0].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.mid_modules.0.blocks.1.block.0.bias'])  # bias
        if 'network.mid_modules.0.blocks.1.block.2.weight' in params_dict:
            self.network.mid_modules[0].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.mid_modules.0.blocks.1.block.2.weight'].T)  # kernel
        if 'network.mid_modules.0.blocks.1.block.2.bias' in params_dict:
            self.network.mid_modules[0].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.mid_modules.0.blocks.1.block.2.bias'])  # bias

        if 'network.mid_modules.0.cond_encoder.0.weight' in params_dict:
            self.network.mid_modules[0].cond_encoder[0].trainable_weights[0].assign(params_dict['network.mid_modules.0.cond_encoder.0.weight'].T)  # kernel
        if 'network.mid_modules.0.cond_encoder.0.bias' in params_dict:
            self.network.mid_modules[0].cond_encoder[0].trainable_weights[1].assign(params_dict['network.mid_modules.0.cond_encoder.0.bias'])  # bias
        if 'network.mid_modules.0.cond_encoder.2.weight' in params_dict:
            self.network.mid_modules[0].cond_encoder[2].trainable_weights[0].assign(params_dict['network.mid_modules.0.cond_encoder.2.weight'].T)  # kernel
        if 'network.mid_modules.0.cond_encoder.2.bias' in params_dict:
            self.network.mid_modules[0].cond_encoder[2].trainable_weights[1].assign(params_dict['network.mid_modules.0.cond_encoder.2.bias'])  # bias
        if 'network.mid_modules.0.cond_encoder.4.weight' in params_dict:
            self.network.mid_modules[0].cond_encoder[4].trainable_weights[0].assign(params_dict['network.mid_modules.0.cond_encoder.4.weight'].T)  # kernel
        if 'network.mid_modules.0.cond_encoder.4.bias' in params_dict:
            self.network.mid_modules[0].cond_encoder[4].trainable_weights[1].assign(params_dict['network.mid_modules.0.cond_encoder.4.bias'])  # bias

        
        

        if 'network.mid_modules.1.blocks.0.block.0.weight' in params_dict:
            self.network.mid_modules[1].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.mid_modules.1.blocks.0.block.0.weight'].T)  # kernel
        if 'network.mid_modules.1.blocks.0.block.0.bias' in params_dict:
            self.network.mid_modules[1].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.mid_modules.1.blocks.0.block.0.bias'])  # bias
        if 'network.mid_modules.1.blocks.0.block.2.weight' in params_dict:
            self.network.mid_modules[1].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.mid_modules.1.blocks.0.block.2.weight'].T)  # kernel
        if 'network.mid_modules.1.blocks.0.block.2.bias' in params_dict:
            self.network.mid_modules[1].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.mid_modules.1.blocks.0.block.2.bias'])  # bias
        if 'network.mid_modules.1.blocks.1.block.0.weight' in params_dict:
            self.network.mid_modules[1].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.mid_modules.1.blocks.1.block.0.weight'].T)  # kernel
        if 'network.mid_modules.1.blocks.1.block.0.bias' in params_dict:
            self.network.mid_modules[1].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.mid_modules.1.blocks.1.block.0.bias'])  # bias
        if 'network.mid_modules.1.blocks.1.block.2.weight' in params_dict:
            self.network.mid_modules[1].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.mid_modules.1.blocks.1.block.2.weight'].T)  # kernel
        if 'network.mid_modules.1.blocks.1.block.2.bias' in params_dict:
            self.network.mid_modules[1].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.mid_modules.1.blocks.1.block.2.bias'])  # bias

        if 'network.mid_modules.1.cond_encoder.0.weight' in params_dict:
            self.network.mid_modules[1].cond_encoder[0].trainable_weights[0].assign(params_dict['network.mid_modules.1.cond_encoder.0.weight'].T)  # kernel
        if 'network.mid_modules.1.cond_encoder.0.bias' in params_dict:
            self.network.mid_modules[1].cond_encoder[0].trainable_weights[1].assign(params_dict['network.mid_modules.1.cond_encoder.0.bias'])  # bias
        if 'network.mid_modules.1.cond_encoder.2.weight' in params_dict:
            self.network.mid_modules[1].cond_encoder[2].trainable_weights[0].assign(params_dict['network.mid_modules.1.cond_encoder.2.weight'].T)  # kernel
        if 'network.mid_modules.1.cond_encoder.2.bias' in params_dict:
            self.network.mid_modules[1].cond_encoder[2].trainable_weights[1].assign(params_dict['network.mid_modules.1.cond_encoder.2.bias'])  # bias
        if 'network.mid_modules.1.cond_encoder.4.weight' in params_dict:
            self.network.mid_modules[1].cond_encoder[4].trainable_weights[0].assign(params_dict['network.mid_modules.1.cond_encoder.4.weight'].T)  # kernel
        if 'network.mid_modules.1.cond_encoder.4.bias' in params_dict:
            self.network.mid_modules[1].cond_encoder[4].trainable_weights[1].assign(params_dict['network.mid_modules.1.cond_encoder.4.bias'])  # bias

        
        




        if 'network.down_modules.0.0.blocks.0.block.0.weight' in params_dict:
            self.network.down_modules[0][0].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.down_modules.0.0.blocks.0.block.0.weight'].T)  # kernel
        if 'network.down_modules.0.0.blocks.0.block.0.bias' in params_dict:
            self.network.down_modules[0][0].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.down_modules.0.0.blocks.0.block.0.bias'])  # bias
        if 'network.down_modules.0.0.blocks.0.block.2.weight' in params_dict:
            self.network.down_modules[0][0].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.down_modules.0.0.blocks.0.block.2.weight'].T)  # kernel
        if 'network.down_modules.0.0.blocks.0.block.2.bias' in params_dict:
            self.network.down_modules[0][0].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.down_modules.0.0.blocks.0.block.2.bias'])  # bias
        if 'network.down_modules.0.0.blocks.1.block.0.weight' in params_dict:
            self.network.down_modules[0][0].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.down_modules.0.0.blocks.1.block.0.weight'].T)  # kernel
        if 'network.down_modules.0.0.blocks.1.block.0.bias' in params_dict:
            self.network.down_modules[0][0].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.down_modules.0.0.blocks.1.block.0.bias'])  # bias
        if 'network.down_modules.0.0.blocks.1.block.2.weight' in params_dict:
            self.network.down_modules[0][0].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.down_modules.0.0.blocks.1.block.2.weight'].T)  # kernel
        if 'network.down_modules.0.0.blocks.1.block.2.bias' in params_dict:
            self.network.down_modules[0][0].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.down_modules.0.0.blocks.1.block.2.bias'])  # bias

        if 'network.down_modules.0.0.cond_encoder.0.weight' in params_dict:
            self.network.down_modules[0][0].cond_encoder[0].trainable_weights[0].assign(params_dict['network.down_modules.0.0.cond_encoder.0.weight'].T)  # kernel
        if 'network.down_modules.0.0.cond_encoder.0.bias' in params_dict:
            self.network.down_modules[0][0].cond_encoder[0].trainable_weights[1].assign(params_dict['network.down_modules.0.0.cond_encoder.0.bias'])  # bias
        if 'network.down_modules.0.0.cond_encoder.2.weight' in params_dict:
            self.network.down_modules[0][0].cond_encoder[2].trainable_weights[0].assign(params_dict['network.down_modules.0.0.cond_encoder.2.weight'].T)  # kernel
        if 'network.down_modules.0.0.cond_encoder.2.bias' in params_dict:
            self.network.down_modules[0][0].cond_encoder[2].trainable_weights[1].assign(params_dict['network.down_modules.0.0.cond_encoder.2.bias'])  # bias
        if 'network.down_modules.0.0.cond_encoder.4.weight' in params_dict:
            self.network.down_modules[0][0].cond_encoder[4].trainable_weights[0].assign(params_dict['network.down_modules.0.0.cond_encoder.4.weight'].T)  # kernel
        if 'network.down_modules.0.0.cond_encoder.4.bias' in params_dict:
            self.network.down_modules[0][0].cond_encoder[4].trainable_weights[1].assign(params_dict['network.down_modules.0.0.cond_encoder.4.bias'])  # bias


        if 'network.down_modules.0.0.residual_conv.weight' in params_dict:
            self.network.down_modules[0][0].residual_conv.trainable_weights[0].assign(params_dict['network.down_modules.0.0.residual_conv.weight'].T)  # kernel
        if 'network.down_modules.0.0.residual_conv.bias' in params_dict:
            self.network.down_modules[0][0].residual_conv.trainable_weights[1].assign(params_dict['network.down_modules.0.0.residual_conv.bias'])  # bias






        if 'network.down_modules.0.1.blocks.0.block.0.weight' in params_dict:
            self.network.down_modules[0][1].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.down_modules.0.1.blocks.0.block.0.weight'].T)  # kernel
        if 'network.down_modules.0.1.blocks.0.block.0.bias' in params_dict:
            self.network.down_modules[0][1].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.down_modules.0.1.blocks.0.block.0.bias'])  # bias
        if 'network.down_modules.0.1.blocks.0.block.2.weight' in params_dict:
            self.network.down_modules[0][1].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.down_modules.0.1.blocks.0.block.2.weight'].T)  # kernel
        if 'network.down_modules.0.1.blocks.0.block.2.bias' in params_dict:
            self.network.down_modules[0][1].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.down_modules.0.1.blocks.0.block.2.bias'])  # bias
        if 'network.down_modules.0.1.blocks.1.block.0.weight' in params_dict:
            self.network.down_modules[0][1].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.down_modules.0.1.blocks.1.block.0.weight'].T)  # kernel
        if 'network.down_modules.0.1.blocks.1.block.0.bias' in params_dict:
            self.network.down_modules[0][1].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.down_modules.0.1.blocks.1.block.0.bias'])  # bias
        if 'network.down_modules.0.1.blocks.1.block.2.weight' in params_dict:
            self.network.down_modules[0][1].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.down_modules.0.1.blocks.1.block.2.weight'].T)  # kernel
        if 'network.down_modules.0.1.blocks.1.block.2.bias' in params_dict:
            self.network.down_modules[0][1].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.down_modules.0.1.blocks.1.block.2.bias'])  # bias

        if 'network.down_modules.0.1.cond_encoder.0.weight' in params_dict:
            self.network.down_modules[0][1].cond_encoder[0].trainable_weights[0].assign(params_dict['network.down_modules.0.1.cond_encoder.0.weight'].T)  # kernel
        if 'network.down_modules.0.1.cond_encoder.0.bias' in params_dict:
            self.network.down_modules[0][1].cond_encoder[0].trainable_weights[1].assign(params_dict['network.down_modules.0.1.cond_encoder.0.bias'])  # bias
        if 'network.down_modules.0.1.cond_encoder.2.weight' in params_dict:
            self.network.down_modules[0][1].cond_encoder[2].trainable_weights[0].assign(params_dict['network.down_modules.0.1.cond_encoder.2.weight'].T)  # kernel
        if 'network.down_modules.0.1.cond_encoder.2.bias' in params_dict:
            self.network.down_modules[0][1].cond_encoder[2].trainable_weights[1].assign(params_dict['network.down_modules.0.1.cond_encoder.2.bias'])  # bias
        if 'network.down_modules.0.1.cond_encoder.4.weight' in params_dict:
            self.network.down_modules[0][1].cond_encoder[4].trainable_weights[0].assign(params_dict['network.down_modules.0.1.cond_encoder.4.weight'].T)  # kernel
        if 'network.down_modules.0.1.cond_encoder.4.bias' in params_dict:
            self.network.down_modules[0][1].cond_encoder[4].trainable_weights[1].assign(params_dict['network.down_modules.0.1.cond_encoder.4.bias'])  # bias


        if 'network.down_modules.0.2.conv.weight' in params_dict:
            self.network.down_modules[0][2].conv.trainable_weights[0].assign(params_dict['network.down_modules.0.2.conv.weight'].T)  # kernel
        if 'network.down_modules.0.2.conv.bias' in params_dict:
            self.network.down_modules[0][2].conv.trainable_weights[1].assign(params_dict['network.down_modules.0.2.conv.bias'])  # bias









        if 'network.down_modules.1.0.blocks.0.block.0.weight' in params_dict:
            self.network.down_modules[1][0].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.down_modules.1.0.blocks.0.block.0.weight'].T)  # kernel
        if 'network.down_modules.1.0.blocks.0.block.0.bias' in params_dict:
            self.network.down_modules[1][0].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.down_modules.1.0.blocks.0.block.0.bias'])  # bias
        if 'network.down_modules.1.0.blocks.0.block.2.weight' in params_dict:
            self.network.down_modules[1][0].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.down_modules.1.0.blocks.0.block.2.weight'].T)  # kernel
        if 'network.down_modules.1.0.blocks.0.block.2.bias' in params_dict:
            self.network.down_modules[1][0].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.down_modules.1.0.blocks.0.block.2.bias'])  # bias
        if 'network.down_modules.1.0.blocks.1.block.0.weight' in params_dict:
            self.network.down_modules[1][0].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.down_modules.1.0.blocks.1.block.0.weight'].T)  # kernel
        if 'network.down_modules.1.0.blocks.1.block.0.bias' in params_dict:
            self.network.down_modules[1][0].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.down_modules.1.0.blocks.1.block.0.bias'])  # bias
        if 'network.down_modules.1.0.blocks.1.block.2.weight' in params_dict:
            self.network.down_modules[1][0].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.down_modules.1.0.blocks.1.block.2.weight'].T)  # kernel
        if 'network.down_modules.1.0.blocks.1.block.2.bias' in params_dict:
            self.network.down_modules[1][0].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.down_modules.1.0.blocks.1.block.2.bias'])  # bias

        if 'network.down_modules.1.0.cond_encoder.0.weight' in params_dict:
            self.network.down_modules[1][0].cond_encoder[0].trainable_weights[0].assign(params_dict['network.down_modules.1.0.cond_encoder.0.weight'].T)  # kernel
        if 'network.down_modules.1.0.cond_encoder.0.bias' in params_dict:
            self.network.down_modules[1][0].cond_encoder[0].trainable_weights[1].assign(params_dict['network.down_modules.1.0.cond_encoder.0.bias'])  # bias
        if 'network.down_modules.1.0.cond_encoder.2.weight' in params_dict:
            self.network.down_modules[1][0].cond_encoder[2].trainable_weights[0].assign(params_dict['network.down_modules.1.0.cond_encoder.2.weight'].T)  # kernel
        if 'network.down_modules.1.0.cond_encoder.2.bias' in params_dict:
            self.network.down_modules[1][0].cond_encoder[2].trainable_weights[1].assign(params_dict['network.down_modules.1.0.cond_encoder.2.bias'])  # bias
        if 'network.down_modules.1.0.cond_encoder.4.weight' in params_dict:
            self.network.down_modules[1][0].cond_encoder[4].trainable_weights[0].assign(params_dict['network.down_modules.1.0.cond_encoder.4.weight'].T)  # kernel
        if 'network.down_modules.1.0.cond_encoder.4.bias' in params_dict:
            self.network.down_modules[1][0].cond_encoder[4].trainable_weights[1].assign(params_dict['network.down_modules.1.0.cond_encoder.4.bias'])  # bias


        if 'network.down_modules.1.0.residual_conv.weight' in params_dict:
            self.network.down_modules[1][0].residual_conv.trainable_weights[0].assign(params_dict['network.down_modules.1.0.residual_conv.weight'].T)  # kernel
        if 'network.down_modules.1.0.residual_conv.bias' in params_dict:
            self.network.down_modules[1][0].residual_conv.trainable_weights[1].assign(params_dict['network.down_modules.1.0.residual_conv.bias'])  # bias




        if 'network.down_modules.1.1.blocks.0.block.0.weight' in params_dict:
            self.network.down_modules[1][1].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.down_modules.1.1.blocks.0.block.0.weight'].T)  # kernel
        if 'network.down_modules.1.1.blocks.0.block.0.bias' in params_dict:
            self.network.down_modules[1][1].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.down_modules.1.1.blocks.0.block.0.bias'])  # bias
        if 'network.down_modules.1.1.blocks.0.block.2.weight' in params_dict:
            self.network.down_modules[1][1].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.down_modules.1.1.blocks.0.block.2.weight'].T)  # kernel
        if 'network.down_modules.1.1.blocks.0.block.2.bias' in params_dict:
            self.network.down_modules[1][1].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.down_modules.1.1.blocks.0.block.2.bias'])  # bias
        if 'network.down_modules.1.1.blocks.1.block.0.weight' in params_dict:
            self.network.down_modules[1][1].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.down_modules.1.1.blocks.1.block.0.weight'].T)  # kernel
        if 'network.down_modules.1.1.blocks.1.block.0.bias' in params_dict:
            self.network.down_modules[1][1].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.down_modules.1.1.blocks.1.block.0.bias'])  # bias
        if 'network.down_modules.1.1.blocks.1.block.2.weight' in params_dict:
            self.network.down_modules[1][1].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.down_modules.1.1.blocks.1.block.2.weight'].T)  # kernel
        if 'network.down_modules.1.1.blocks.1.block.2.bias' in params_dict:
            self.network.down_modules[1][1].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.down_modules.1.1.blocks.1.block.2.bias'])  # bias

        if 'network.down_modules.1.1.cond_encoder.0.weight' in params_dict:
            self.network.down_modules[1][1].cond_encoder[0].trainable_weights[0].assign(params_dict['network.down_modules.1.1.cond_encoder.0.weight'].T)  # kernel
        if 'network.down_modules.1.1.cond_encoder.0.bias' in params_dict:
            self.network.down_modules[1][1].cond_encoder[0].trainable_weights[1].assign(params_dict['network.down_modules.1.1.cond_encoder.0.bias'])  # bias
        if 'network.down_modules.1.1.cond_encoder.2.weight' in params_dict:
            self.network.down_modules[1][1].cond_encoder[2].trainable_weights[0].assign(params_dict['network.down_modules.1.1.cond_encoder.2.weight'].T)  # kernel
        if 'network.down_modules.1.1.cond_encoder.2.bias' in params_dict:
            self.network.down_modules[1][1].cond_encoder[2].trainable_weights[1].assign(params_dict['network.down_modules.1.1.cond_encoder.2.bias'])  # bias
        if 'network.down_modules.1.1.cond_encoder.4.weight' in params_dict:
            self.network.down_modules[1][1].cond_encoder[4].trainable_weights[0].assign(params_dict['network.down_modules.1.1.cond_encoder.4.weight'].T)  # kernel
        if 'network.down_modules.1.1.cond_encoder.4.bias' in params_dict:
            self.network.down_modules[1][1].cond_encoder[4].trainable_weights[1].assign(params_dict['network.down_modules.1.1.cond_encoder.4.bias'])  # bias










        if 'network.up_modules.0.0.blocks.0.block.0.weight' in params_dict:
            self.network.up_modules[0][0].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.up_modules.0.0.blocks.0.block.0.weight'].T)  # kernel
        if 'network.up_modules.0.0.blocks.0.block.0.bias' in params_dict:
            self.network.up_modules[0][0].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.up_modules.0.0.blocks.0.block.0.bias'])  # bias
        if 'network.up_modules.0.0.blocks.0.block.2.weight' in params_dict:
            self.network.up_modules[0][0].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.up_modules.0.0.blocks.0.block.2.weight'].T)  # kernel
        if 'network.up_modules.0.0.blocks.0.block.2.bias' in params_dict:
            self.network.up_modules[0][0].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.up_modules.0.0.blocks.0.block.2.bias'])  # bias
        if 'network.up_modules.0.0.blocks.1.block.0.weight' in params_dict:
            self.network.up_modules[0][0].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.up_modules.0.0.blocks.1.block.0.weight'].T)  # kernel
        if 'network.up_modules.0.0.blocks.1.block.0.bias' in params_dict:
            self.network.up_modules[0][0].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.up_modules.0.0.blocks.1.block.0.bias'])  # bias
        if 'network.up_modules.0.0.blocks.1.block.2.weight' in params_dict:
            self.network.up_modules[0][0].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.up_modules.0.0.blocks.1.block.2.weight'].T)  # kernel
        if 'network.up_modules.0.0.blocks.1.block.2.bias' in params_dict:
            self.network.up_modules[0][0].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.up_modules.0.0.blocks.1.block.2.bias'])  # bias

        if 'network.up_modules.0.0.cond_encoder.0.weight' in params_dict:
            self.network.up_modules[0][0].cond_encoder[0].trainable_weights[0].assign(params_dict['network.up_modules.0.0.cond_encoder.0.weight'].T)  # kernel
        if 'network.up_modules.0.0.cond_encoder.0.bias' in params_dict:
            self.network.up_modules[0][0].cond_encoder[0].trainable_weights[1].assign(params_dict['network.up_modules.0.0.cond_encoder.0.bias'])  # bias
        if 'network.up_modules.0.0.cond_encoder.2.weight' in params_dict:
            self.network.up_modules[0][0].cond_encoder[2].trainable_weights[0].assign(params_dict['network.up_modules.0.0.cond_encoder.2.weight'].T)  # kernel
        if 'network.up_modules.0.0.cond_encoder.2.bias' in params_dict:
            self.network.up_modules[0][0].cond_encoder[2].trainable_weights[1].assign(params_dict['network.up_modules.0.0.cond_encoder.2.bias'])  # bias
        if 'network.up_modules.0.0.cond_encoder.4.weight' in params_dict:
            self.network.up_modules[0][0].cond_encoder[4].trainable_weights[0].assign(params_dict['network.up_modules.0.0.cond_encoder.4.weight'].T)  # kernel
        if 'network.up_modules.0.0.cond_encoder.4.bias' in params_dict:
            self.network.up_modules[0][0].cond_encoder[4].trainable_weights[1].assign(params_dict['network.up_modules.0.0.cond_encoder.4.bias'])  # bias


        if 'network.up_modules.0.0.residual_conv.weight' in params_dict:
            self.network.up_modules[0][0].residual_conv.trainable_weights[0].assign(params_dict['network.up_modules.0.0.residual_conv.weight'].T)  # kernel
        if 'network.up_modules.0.0.residual_conv.bias' in params_dict:
            self.network.up_modules[0][0].residual_conv.trainable_weights[1].assign(params_dict['network.up_modules.0.0.residual_conv.bias'])  # bias



        if 'network.up_modules.0.1.blocks.0.block.0.weight' in params_dict:
            self.network.up_modules[0][1].blocks[0].block[0].trainable_weights[0].assign(params_dict['network.up_modules.0.1.blocks.0.block.0.weight'].T)  # kernel
        if 'network.up_modules.0.1.blocks.0.block.0.bias' in params_dict:
            self.network.up_modules[0][1].blocks[0].block[0].trainable_weights[1].assign(params_dict['network.up_modules.0.1.blocks.0.block.0.bias'])  # bias
        if 'network.up_modules.0.1.blocks.0.block.2.weight' in params_dict:
            self.network.up_modules[0][1].blocks[0].block[2].trainable_weights[0].assign(params_dict['network.up_modules.0.1.blocks.0.block.2.weight'].T)  # kernel
        if 'network.up_modules.0.1.blocks.0.block.2.bias' in params_dict:
            self.network.up_modules[0][1].blocks[0].block[2].trainable_weights[1].assign(params_dict['network.up_modules.0.1.blocks.0.block.2.bias'])  # bias
        if 'network.up_modules.0.1.blocks.1.block.0.weight' in params_dict:
            self.network.up_modules[0][1].blocks[1].block[0].trainable_weights[0].assign(params_dict['network.up_modules.0.1.blocks.1.block.0.weight'].T)  # kernel
        if 'network.up_modules.0.1.blocks.1.block.0.bias' in params_dict:
            self.network.up_modules[0][1].blocks[1].block[0].trainable_weights[1].assign(params_dict['network.up_modules.0.1.blocks.1.block.0.bias'])  # bias
        if 'network.up_modules.0.1.blocks.1.block.2.weight' in params_dict:
            self.network.up_modules[0][1].blocks[1].block[2].trainable_weights[0].assign(params_dict['network.up_modules.0.1.blocks.1.block.2.weight'].T)  # kernel
        if 'network.up_modules.0.1.blocks.1.block.2.bias' in params_dict:
            self.network.up_modules[0][1].blocks[1].block[2].trainable_weights[1].assign(params_dict['network.up_modules.0.1.blocks.1.block.2.bias'])  # bias

        if 'network.up_modules.0.1.cond_encoder.0.weight' in params_dict:
            self.network.up_modules[0][1].cond_encoder[0].trainable_weights[0].assign(params_dict['network.up_modules.0.1.cond_encoder.0.weight'].T)  # kernel
        if 'network.up_modules.0.1.cond_encoder.0.bias' in params_dict:
            self.network.up_modules[0][1].cond_encoder[0].trainable_weights[1].assign(params_dict['network.up_modules.0.1.cond_encoder.0.bias'])  # bias
        if 'network.up_modules.0.1.cond_encoder.2.weight' in params_dict:
            self.network.up_modules[0][1].cond_encoder[2].trainable_weights[0].assign(params_dict['network.up_modules.0.1.cond_encoder.2.weight'].T)  # kernel
        if 'network.up_modules.0.1.cond_encoder.2.bias' in params_dict:
            self.network.up_modules[0][1].cond_encoder[2].trainable_weights[1].assign(params_dict['network.up_modules.0.1.cond_encoder.2.bias'])  # bias
        if 'network.up_modules.0.1.cond_encoder.4.weight' in params_dict:
            self.network.up_modules[0][1].cond_encoder[4].trainable_weights[0].assign(params_dict['network.up_modules.0.1.cond_encoder.4.weight'].T)  # kernel
        if 'network.up_modules.0.1.cond_encoder.4.bias' in params_dict:
            self.network.up_modules[0][1].cond_encoder[4].trainable_weights[1].assign(params_dict['network.up_modules.0.1.cond_encoder.4.bias'])  # bias


        if 'network.up_modules.0.2.conv.weight' in params_dict:
            self.network.up_modules[0][2].conv.trainable_weights[0].assign(params_dict['network.up_modules.0.2.conv.weight'].T)  # kernel
        if 'network.up_modules.0.2.conv.bias' in params_dict:
            self.network.up_modules[0][2].conv.trainable_weights[1].assign(params_dict['network.up_modules.0.2.conv.bias'])  # bias




        if 'network.final_conv.0.block.0.weight':
            self.network.final_conv[0].block[0].trainable_weights[0].assign(params_dict['network.final_conv.0.block.0.weight'].T)  # kernel
        if 'network.final_conv.0.block.0.bias':
            self.network.final_conv[0].block[0].trainable_weights[1].assign(params_dict['network.final_conv.0.block.0.bias'])  # bias

        if 'network.final_conv.0.block.2.weight':
            self.network.final_conv[0].block[2].trainable_weights[0].assign(params_dict['network.final_conv.0.block.2.weight'].T)  # kernel
        if 'network.final_conv.0.block.2.bias':
            self.network.final_conv[0].block[2].trainable_weights[1].assign(params_dict['network.final_conv.0.block.2.bias'])  # bias


        if 'network.final_conv.1.weight':
            self.network.final_conv[1].trainable_weights[0].assign(params_dict['network.final_conv.1.weight'].T)  # kernel
        if 'network.final_conv.1.bias':
            self.network.final_conv[1].trainable_weights[1].assign(params_dict['network.final_conv.1.bias'])  # bias






    def load_pickle_diffusion_mlp_img(self, network_path):
        pkl_file_path = network_path.replace('.pt', '_ema.pkl')

        print("pkl_file_path = ", pkl_file_path)

        import pickle
        with open(pkl_file_path, 'rb') as file:
            params_dict = pickle.load(file)


        if OUTPUT_VARIABLES:
            print("params_dict = ", params_dict)





        if 'network.backbone.vit.pos_embed' in params_dict:
            self.network.backbone.vit.pos_embed = nn_Parameter( torch_tensor(params_dict['network.backbone.vit.pos_embed']) )
            
        if 'network.backbone.vit.patch_embed.embed.0.weight' in params_dict:
            self.network.backbone.vit.patch_embed.embed[0].trainable_weights[0].assign(params_dict['network.backbone.vit.patch_embed.embed.0.weight'].T)  # kernel
        if 'network.backbone.vit.patch_embed.embed.0.bias' in params_dict:
            self.network.backbone.vit.patch_embed.embed[0].trainable_weights[1].assign(params_dict['network.backbone.vit.patch_embed.embed.0.bias'])  # bias

        if 'network.backbone.vit.patch_embed.embed.3.weight' in params_dict:
            self.network.backbone.vit.patch_embed.embed[3].trainable_weights[0].assign(params_dict['network.backbone.vit.patch_embed.embed.3.weight'].T)  # kernel
        if 'network.backbone.vit.patch_embed.embed.3.bias' in params_dict:
            self.network.backbone.vit.patch_embed.embed[3].trainable_weights[1].assign(params_dict['network.backbone.vit.patch_embed.embed.3.bias'])  # bias




        if 'network.backbone.vit.net.0.layer_norm1.weight' in params_dict:
            self.network.backbone.vit.net[0].layer_norm1.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.layer_norm1.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.layer_norm1.bias' in params_dict:
            self.network.backbone.vit.net[0].layer_norm1.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.layer_norm1.bias'])  # bias

        if 'network.backbone.vit.net.0.mha.qkv_proj.weight' in params_dict:
            self.network.backbone.vit.net[0].mha.qkv_proj.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.mha.qkv_proj.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.mha.qkv_proj.bias' in params_dict:
            self.network.backbone.vit.net[0].mha.qkv_proj.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.mha.qkv_proj.bias'])  # bias


        if 'network.backbone.vit.net.0.mha.out_proj.weight' in params_dict:
            self.network.backbone.vit.net[0].mha.out_proj.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.mha.out_proj.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.mha.out_proj.bias' in params_dict:
            self.network.backbone.vit.net[0].mha.out_proj.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.mha.out_proj.bias'])  # bias

        if 'network.backbone.vit.net.0.layer_norm2.weight' in params_dict:
            self.network.backbone.vit.net[0].layer_norm2.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.layer_norm2.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.layer_norm2.bias' in params_dict:
            self.network.backbone.vit.net[0].layer_norm2.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.layer_norm2.bias'])  # bias


        if 'network.backbone.vit.net.0.linear1.weight' in params_dict:
            self.network.backbone.vit.net[0].linear1.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.linear1.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.linear1.bias' in params_dict:
            self.network.backbone.vit.net[0].linear1.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.linear1.bias'])  # bias


        if 'network.backbone.vit.net.0.linear2.weight' in params_dict:
            self.network.backbone.vit.net[0].linear2.trainable_weights[0].assign(params_dict['network.backbone.vit.net.0.linear2.weight'].T)  # kernel
        if 'network.backbone.vit.net.0.linear2.bias' in params_dict:
            self.network.backbone.vit.net[0].linear2.trainable_weights[1].assign(params_dict['network.backbone.vit.net.0.linear2.bias'])  # bias


        if 'network.backbone.vit.norm.weight' in params_dict:
            self.network.backbone.vit.norm.trainable_weights[0].assign(params_dict['network.backbone.vit.norm.weight'].T)  # kernel
        if 'network.backbone.vit.norm.bias' in params_dict:
            self.network.backbone.vit.norm.trainable_weights[1].assign(params_dict['network.backbone.vit.norm.bias'])  # bias



        if 'network.compress.weight' in params_dict:
            if isinstance(self.network.compress.weight, tf.Variable):       
                self.network.compress.weight = nn_Parameter(
                    torch_tensor(params_dict['network.compress.weight']), requires_grad=False
                )
            else:
                self.network.compress.weight.trainable_weights[0].assign(params_dict['network.compress.weight'])  # kernel

        if 'network.compress.input_proj.0.weight' in params_dict:
            self.network.compress.input_proj[0].trainable_weights[0].assign(params_dict['network.compress.input_proj.0.weight'].T)  # kernel
        if 'network.compress.input_proj.0.bias' in params_dict:
            self.network.compress.input_proj[0].trainable_weights[1].assign(params_dict['network.compress.input_proj.0.bias'])  # bias

        if 'network.compress.input_proj.1.weight' in params_dict:
            self.network.compress.input_proj[1].trainable_weights[0].assign(params_dict['network.compress.input_proj.1.weight'].T)  # kernel
        if 'network.compress.input_proj.1.bias' in params_dict:
            self.network.compress.input_proj[1].trainable_weights[1].assign(params_dict['network.compress.input_proj.1.bias'])  # bias






        if 'network.compress1.weight' in params_dict:
            if isinstance(self.network.compress1.weight, tf.Variable):       
                self.network.compress1.weight = nn_Parameter(
                    torch_tensor(params_dict['network.compress1.weight']), requires_grad=False
                )
            else:
                self.network.compress1.weight.trainable_weights[0].assign(params_dict['network.compress1.weight'])  # kernel

        if 'network.compress1.input_proj.0.weight' in params_dict:
            self.network.compress1.input_proj[0].trainable_weights[0].assign(params_dict['network.compress1.input_proj.0.weight'].T)  # kernel
        if 'network.compress1.input_proj.0.bias' in params_dict:
            self.network.compress1.input_proj[0].trainable_weights[1].assign(params_dict['network.compress1.input_proj.0.bias'])  # bias

        if 'network.compress1.input_proj.1.weight' in params_dict:
            self.network.compress1.input_proj[1].trainable_weights[0].assign(params_dict['network.compress1.input_proj.1.weight'].T)  # kernel
        if 'network.compress1.input_proj.1.bias' in params_dict:
            self.network.compress1.input_proj[1].trainable_weights[1].assign(params_dict['network.compress1.input_proj.1.bias'])  # bias








        if 'network.compress2.weight' in params_dict:
            if isinstance(self.network.compress2.weight, tf.Variable):       
                self.network.compress2.weight = nn_Parameter(
                    torch_tensor(params_dict['network.compress2.weight']), requires_grad=False
                )
            else:
                self.network.compress2.weight.trainable_weights[0].assign(params_dict['network.compress2.weight'])  # kernel


        if 'network.compress2.input_proj.0.weight' in params_dict:
            self.network.compress2.input_proj[0].trainable_weights[0].assign(params_dict['network.compress2.input_proj.0.weight'].T)  # kernel
        if 'network.compress2.input_proj.0.bias' in params_dict:
            self.network.compress2.input_proj[0].trainable_weights[1].assign(params_dict['network.compress2.input_proj.0.bias'])  # bias

        if 'network.compress2.input_proj.1.weight' in params_dict:
            self.network.compress2.input_proj[1].trainable_weights[0].assign(params_dict['network.compress2.input_proj.1.weight'].T)  # kernel
        if 'network.compress2.input_proj.1.bias' in params_dict:
            self.network.compress2.input_proj[1].trainable_weights[1].assign(params_dict['network.compress2.input_proj.1.bias'])  # bias











        if 'network.time_embedding.1.weight' in params_dict:
            self.network.time_embedding[1].trainable_weights[0].assign(params_dict['network.time_embedding.1.weight'].T)  # kernel

        if 'network.time_embedding.1.bias' in params_dict:
            self.network.time_embedding[1].trainable_weights[1].assign(params_dict['network.time_embedding.1.bias'])     # bias

        if 'network.time_embedding.3.weight' in params_dict:
            self.network.time_embedding[3].trainable_weights[0].assign(params_dict['network.time_embedding.3.weight'].T)  # kernel

        if 'network.time_embedding.3.bias' in params_dict:
            self.network.time_embedding[3].trainable_weights[1].assign(params_dict['network.time_embedding.3.bias'])     # bias






        if 'network.mlp_mean.layers.0.weight' in params_dict:
            self.network.mlp_mean.my_layers[0].trainable_weights[0].assign(params_dict['network.mlp_mean.layers.0.weight'].T)  # kernel
        if 'network.mlp_mean.layers.0.bias' in params_dict:
            self.network.mlp_mean.my_layers[0].trainable_weights[1].assign(params_dict['network.mlp_mean.layers.0.bias'])     # bias

        if 'network.mlp_mean.layers.1.l1.weight' in params_dict:
            self.network.mlp_mean.my_layers[1].l1.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.1.l1.weight'].T)  # kernel
        if 'network.mlp_mean.layers.1.l1.bias' in params_dict:
            self.network.mlp_mean.my_layers[1].l1.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.1.l1.bias'])     # bias

        if 'network.mlp_mean.layers.1.l2.weight' in params_dict:
            self.network.mlp_mean.my_layers[1].l2.trainable_weights[0].assign(params_dict['network.mlp_mean.layers.1.l2.weight'].T)  # kernel
        if 'network.mlp_mean.layers.1.l2.bias' in params_dict:
            self.network.mlp_mean.my_layers[1].l2.trainable_weights[1].assign(params_dict['network.mlp_mean.layers.1.l2.bias'])     # bias

        if 'network.mlp_mean.layers.2.weight' in params_dict:
            self.network.mlp_mean.my_layers[2].trainable_weights[0].assign(params_dict['network.mlp_mean.layers.2.weight'].T)  # kernel
        if 'network.mlp_mean.layers.2.bias' in params_dict:
            self.network.mlp_mean.my_layers[2].trainable_weights[1].assign(params_dict['network.mlp_mean.layers.2.bias'])     # bias






















    def output_weights(self, actor = None):

        if actor == None:
            self.actor = self.network
        else:
            self.actor = actor
            

        # Time embedding layer 1
        print("actor.time_embedding[1].trainable_weights[0] (kernel):")
        print(self.actor.time_embedding[1].trainable_weights[0].numpy())
        print("actor.time_embedding[1].trainable_weights[1] (bias):")
        print(self.actor.time_embedding[1].trainable_weights[1].numpy())

        # Time embedding layer 3
        print("actor.time_embedding[3].trainable_weights[0] (kernel):")
        print(self.actor.time_embedding[3].trainable_weights[0].numpy())
        print("actor.time_embedding[3].trainable_weights[1] (bias):")
        print(self.actor.time_embedding[3].trainable_weights[1].numpy())



        if self.actor.cond_mlp:
            print("self.actor.cond_mlp.moduleList[0].trainable_weights[0].numpy() = ")
            print(self.actor.cond_mlp.moduleList[0].trainable_weights[0].numpy())

            print("self.network.cond_mlp.moduleList[0].trainable_weights[1].numpy() = ")
            print(self.network.cond_mlp.moduleList[0].trainable_weights[1].numpy())
                
            print("self.network.cond_mlp.moduleList[1].trainable_weights[0].numpy() = ")
            print(self.network.cond_mlp.moduleList[1].trainable_weights[0].numpy())
                
            print("self.network.cond_mlp.moduleList[1].trainable_weights[1].numpy() = ")
            print(self.network.cond_mlp.moduleList[1].trainable_weights[1].numpy())







        # MLP mean layer 0
        print("actor.mlp_mean.my_layers[0].trainable_weights[0] (kernel):")
        print(self.actor.mlp_mean.my_layers[0].trainable_weights[0].numpy())
        print("actor.mlp_mean.my_layers[0].trainable_weights[1] (bias):")
        print(self.actor.mlp_mean.my_layers[0].trainable_weights[1].numpy())

        # MLP mean layer 1.l1
        print("actor.mlp_mean.my_layers[1].l1.trainable_weights[0] (kernel):")
        print(self.actor.mlp_mean.my_layers[1].l1.trainable_weights[0].numpy())
        print("actor.mlp_mean.my_layers[1].l1.trainable_weights[1] (bias):")
        print(self.actor.mlp_mean.my_layers[1].l1.trainable_weights[1].numpy())

        # MLP mean layer 1.l2
        print("actor.mlp_mean.my_layers[1].l2.trainable_weights[0] (kernel):")
        print(self.actor.mlp_mean.my_layers[1].l2.trainable_weights[0].numpy())
        print("actor.mlp_mean.my_layers[1].l2.trainable_weights[1] (bias):")
        print(self.actor.mlp_mean.my_layers[1].l2.trainable_weights[1].numpy())

        # MLP mean layer 2
        print("actor.mlp_mean.my_layers[2].trainable_weights[0] (kernel):")
        print(self.actor.mlp_mean.my_layers[2].trainable_weights[0].numpy())
        print("actor.mlp_mean.my_layers[2].trainable_weights[1] (bias):")
        print(self.actor.mlp_mean.my_layers[2].trainable_weights[1].numpy())








