"""
Policy gradient for Gaussian policy

"""

import tensorflow as tf

from copy import deepcopy
import logging
from model.common.gaussian import GaussianModel

from util.torch_to_tf import torch_no_grad, torch_tensor_view, torch_mean

from util.config import METHOD_NAME


class VPG_Gaussian(GaussianModel):

    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):

        print("gaussian_vpg.py: VPG_Gaussian.__init__()")

        super().__init__(network=actor, **kwargs)

        # Value function for obs - simple MLP
        self.critic = critic
        # .to(self.device)

        # # Re-name network to actor
        # self.actor_ft = actor

        # # # Save a copy of original actor
        # # self.actor = deepcopy(actor)



        # Re-name network to actor
        self.actor_ft = self.network


        # self.build_actor(self.actor_ft)
        if "ViT" in METHOD_NAME:            
            self.build_actor_vision(self.actor_ft)
        else:
            self.build_actor(self.actor_ft)

        
        self.actor = tf.keras.models.clone_model(self.actor_ft)

        # self.build_actor(self.actor)
        if "ViT" in METHOD_NAME:            
            self.build_actor_vision(self.actor)
        else:
            self.build_actor(self.actor)


        self.actor.set_weights(self.actor_ft.get_weights())





        # for param in self.actor.parameters():
        #     param.requires_grad = False

        for layer in self.actor.layers:
            layer.trainable = False



    # ---------- Sampling ----------#

    # @torch.no_grad()
    # @tf.function
    def call(
        self,
        cond,
        deterministic=False,
        use_base_policy=False,
    ):
        with torch_no_grad() as tape:

            print("gaussian_vpg.py: VPG_Gaussian.call()")

            return super().call(
                cond=cond,
                deterministic=deterministic,
                network_override=self.actor if use_base_policy else None,
            )
        

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        actions,
        use_base_policy=False,
        training = True
    ):

        print("gaussian_vpg.py: VPG_Gaussian.get_logprobs()")

        B = tf.shape(actions)[0]

        dist = self.forward_train(
            training,
            cond,
            deterministic=False,
            network_override=self.actor if use_base_policy else None,
        )

        # Compute log probability
        log_prob = dist.log_prob( torch_tensor_view(actions, [B, -1]) )
        log_prob = torch_mean(log_prob, -1)

        # Compute entropy and standard deviation
        entropy = torch_mean(dist.entropy())
        # std = torch_mean(dist.stddev())
        std = torch_mean(dist.scale)

        return log_prob, entropy, std

    def loss_ori(self, obs, actions, reward):

        print("gaussian_vpg.py: VPG_Gaussian.loss()")

        raise NotImplementedError
