"""
Imitation Bootstrapped Reinforcement Learning (IBRL) for Gaussian policy.

"""

import tensorflow as tf
import numpy as np

import logging
from copy import deepcopy

from model.common.gaussian import GaussianModel

log = logging.getLogger(__name__)


from util.torch_to_tf import torch_mean, torch_where, torch_softmax, torch_stack, torch_multinomial, torch_func_functional_call, \
    torch_min, torch_func_stack_module_state, torch_vmap

from util.torch_to_tf import torch_no_grad



class IBRL_Gaussian(GaussianModel):
    def __init__(
        self,
        actor,
        critic,
        n_critics,
        soft_action_sample=False,
        soft_action_sample_beta=10,
        **kwargs,
    ):

        print("gaussian_ibrl.py: IBRL_Gaussian.__init__()")

        super().__init__(network=actor, **kwargs)
        self.soft_action_sample = soft_action_sample
        self.soft_action_sample_beta = soft_action_sample_beta

        # Set up target actor
        self.target_actor = deepcopy(actor)



        # Frozen pre-trained policy
        self.bc_policy = deepcopy(actor)
        self.bc_policy.trainable = False


        # initialize critic networks
        self.critic_networks = [
            deepcopy(critic) for _ in range(n_critics)
        ]
        # self.critic_networks = nn.ModuleList(self.critic_networks)

        # initialize target networks
        self.target_networks = [
            deepcopy(critic) for _ in range(n_critics)
        ]
        # self.target_networks = nn.ModuleList(self.target_networks)

        # Construct a "stateless" version of one of the models. It is "stateless" in the sense that the parameters are meta Tensors and do not have storage.
        base_model = deepcopy(self.critic_networks[0])
        self.base_model = base_model.to("meta")



        self.ensemble_params, self.ensemble_buffers = torch_func_stack_module_state(
            self.critic_networks
        )



        self.ensemble_params = [critic.trainable_weights for critic in self.critic_networks]




    def critic_wrapper(self, params, buffers, data):
        """for vmap"""

        print("gaussian_ibrl.py: IBRL_Gaussian.critic_wrapper()")

        # return torch.func.functional_call(self.base_model, (params, buffers), data)
        print("params = ", params)
        print("buffers = ", buffers)

        return torch_func_functional_call(self.base_model, (params, buffers), data)



    def get_random_indices(self, sz=None, num_ind=2):
        """get num_ind random indices from a set of size sz (used for getting critic targets)"""

        print("gaussian_ibrl.py: IBRL_Gaussian.get_random_indices()")

        if sz is None:
            sz = len(self.critic_networks)

        perm = tf.random.shuffle(tf.range(sz))

        ind = perm[:num_ind]
        # .to(self.device)
        return ind

    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        rewards,
        terminated,
        gamma,
    ):

        print("gaussian_ibrl.py: IBRL_Gaussian.loss_critic()")

        # get random critic index
        q1_ind, q2_ind = self.get_random_indices()

        # with tf.GradientTape(persistent=True) as tape:
        with torch_no_grad() as tape:
            next_actions_bc = super().call(
                cond=next_obs,
                deterministic=True,
                network_override=self.bc_policy,
            )

            next_actions_rl = super().call(
                cond=next_obs,
                deterministic=False,
                network_override=self.target_actor,
            )

            # BC Q value
            next_q1_bc = self.target_networks[q1_ind](next_obs, next_actions_bc)
            next_q2_bc = self.target_networks[q2_ind](next_obs, next_actions_bc)
            next_q_bc = tf.minimum(next_q1_bc, next_q2_bc)

            # RL Q value
            next_q1_rl = self.target_networks[q1_ind](next_obs, next_actions_rl)
            next_q2_rl = self.target_networks[q2_ind](next_obs, next_actions_rl)
            next_q_rl = tf.minimum(next_q1_rl, next_q2_rl)

            # Target Q value
            next_q = tf.where(next_q_bc > next_q_rl, next_q_bc, next_q_rl)
            target_q = rewards + gamma * (1 - terminated) * next_q

        # Current Q value
        current_q_list = [
            critic(obs, actions) for critic in self.critic_networks
        ]
        current_q = tf.stack(current_q_list, axis=0)  # Shape: (n_critics, batch_size)
        loss_critic = tf.reduce_mean((current_q - target_q[None, :]) ** 2)

        # # run all critics in batch
        # current_q = torch_vmap( self.critic_wrapper, in_dims=(0, 0, None) )(
        #     self.ensemble_params, self.ensemble_buffers, (obs, actions)
        # )  # (n_critics, B)

        # run all critics in batch
        current_q = torch_vmap( self.critic_wrapper, self.ensemble_params, self.ensemble_buffers, (obs, actions), in_dims=(0, 0, None) )  # (n_critics, B)


        loss_critic = torch_mean((current_q - target_q[None]) ** 2)
        return loss_critic

    def loss_actor(self, obs):

        print("gaussian_ibrl.py: IBRL_Gaussian.loss_actor()")

        action = super().call(
            obs,
            deterministic=False,
            reparameterize=True,
        )  # use online policy only, also IBRL does not use tanh squashing
        
        # current_q = torch_vmap(self.critic_wrapper, in_dims=(0, 0, None))(
        #     self.ensemble_params, self.ensemble_buffers, (obs, action)
        # )  # (n_critics, B)

        current_q = torch_vmap(self.critic_wrapper, self.ensemble_params, self.ensemble_buffers, (obs, action), in_dims=(0, 0, None) )  # (n_critics, B)

        # current_q = current_q.min(
        #     dim=0
        # ).values  # unlike RLPD, IBRL uses the min Q value for actor update

        current_q = torch_min( current_q, dim=0 ).values  # unlike RLPD, IBRL uses the min Q value for actor update
        
        loss_actor = -torch_mean(current_q)
        return loss_actor

















    def update_target_critic(self, tau):
        """Update the target critic using soft updates"""
        print("gaussian_ibrl.py: IBRL_Gaussian.update_target_critic()")

        for target_ind, target_critic in enumerate(self.target_networks):
            for target_param_name, target_param in target_critic.trainable_variables:
                source_param = self.ensemble_params[target_param_name][target_ind]
                updated_value = target_param * (1.0 - tau) + source_param * tau
                target_param.assign(updated_value)






    def update_target_actor(self, tau):
        """Update the target actor using soft updates"""
        print("gaussian_ibrl.py: IBRL_Gaussian.update_target_actor()")

        for target_param, source_param in zip(
            self.target_actor.trainable_variables, self.network.trainable_variables
        ):
            updated_value = target_param * (1.0 - tau) + source_param * tau
            target_param.assign(updated_value)


























    # ---------- Sampling ----------#

    def call(
        self,
        cond,
        deterministic=False,
        reparameterize=False,
    ):
        """use both pre-trained and online policies"""

        print("gaussian_ibrl.py: IBRL_Gaussian.forward()")

        q1_ind, q2_ind = self.get_random_indices()

        # sample an action from the BC policy
        bc_action = super().call(
            cond=cond,
            deterministic=True,
            network_override=self.bc_policy,
        )

        # sample an action from the RL policy
        rl_action = super().call(
            cond=cond,
            deterministic=deterministic,
            reparameterize=reparameterize,
        )

        # compute Q value of BC policy
        q_bc_1 = self.critic_networks[q1_ind](cond, bc_action)
        q_bc_2 = self.critic_networks[q2_ind](cond, bc_action)
        q_bc = tf.minimum(q_bc_1, q_bc_2)

        # compute Q value of RL policy
        q_rl_1 = self.critic_networks[q1_ind](cond, rl_action)
        q_rl_2 = self.critic_networks[q2_ind](cond, rl_action)
        q_rl = tf.minimum(q_rl_1, q_rl_2)

        # soft sample or greedy
        if deterministic or not self.soft_action_sample:
            action = torch_where(
                (q_bc > q_rl)[:, None, None],
                bc_action,
                rl_action,
            )

            action = tf.where(q_bc > q_rl, bc_action, rl_action)

        else:
            # compute the Q weights with probability proportional to exp(\beta * Q(a))
            qw_bc = tf.exp(q_bc * self.soft_action_sample_beta)
            qw_rl = tf.exp(q_rl * self.soft_action_sample_beta)


            q_weights = torch_softmax(
                torch_stack([qw_bc, qw_rl], dim=-1),
                dim=-1,
            )

            # sample according to the weights
            q_indices = torch_multinomial(q_weights, 1)
            action = torch_where(
                (q_indices == 0)[:, None],
                bc_action,
                rl_action,
            )


        return action