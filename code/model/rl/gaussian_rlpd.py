"""
Reinforcement learning with prior data (RLPD) for Gaussian policy.

Use ensemble of critics.

"""

from util.torch_to_tf import torch_mean, torch_min, torch_randperm, torch_func_functional_call, torch_func_stack_module_state, torch_vmap

from util.torch_to_tf import torch_no_grad, nn_ModuleList, nn_Sequential

import logging
from copy import deepcopy

from model.common.gaussian import GaussianModel

log = logging.getLogger(__name__)


class RLPD_Gaussian(GaussianModel):
    def __init__(
        self,
        actor,
        critic,
        n_critics,
        backup_entropy=False,
        **kwargs,
    ):

        print("gaussian_rlpd.py: RLPD_Gaussian.__init__()")

        super().__init__(network=actor, **kwargs)
        self.n_critics = n_critics
        self.backup_entropy = backup_entropy

        # initialize critic networks
        self.critic_networks = [
            deepcopy(critic)
            for _ in range(n_critics)
        ]

        self.critic_networks = nn_Sequential(self.critic_networks)

        # initialize target networks
        self.target_networks = [
            deepcopy(critic)
            for _ in range(n_critics)
        ]

        self.target_networks = nn_Sequential(self.target_networks)

        # Construct a "stateless" version of one of the models. It is "stateless" in the sense that the parameters are meta Tensors and do not have storage.
        base_model = deepcopy(self.critic_networks[0])
        self.base_model = base_model

        self.ensemble_params, self.ensemble_buffers = torch_func_stack_module_state(
            self.critic_networks
        )

    def critic_wrapper(self, params, buffers, data):
        """for vmap"""

        print("gaussian_rlpd.py: RLPD_Gaussian.critic_wrapper()")

        return torch_func_functional_call(self.base_model, (params, buffers), data)

    def get_random_indices(self, sz=None, num_ind=2):
        """get num_ind random indices from a set of size sz (used for getting critic targets)"""

        print("gaussian_rlpd.py: RLPD_Gaussian.get_random_indices()")

        if sz is None:
            sz = len(self.critic_networks)
        perm = torch_randperm(sz)
        ind = perm[:num_ind]

        return ind

    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        rewards,
        terminated,
        gamma,
        alpha,
    ):

        print("gaussian_rlpd.py: RLPD_Gaussian.loss_critic()")

        # get random critic index
        q1_ind, q2_ind = self.get_random_indices()

        with torch_no_grad() as tape:
            next_actions, next_logprobs = self.call(
                cond=next_obs,
                deterministic=False,
                get_logprob=True,
            )
            next_q1 = self.target_networks[q1_ind](next_obs, next_actions)
            next_q2 = self.target_networks[q2_ind](next_obs, next_actions)
            next_q = torch_min(next_q1, other=next_q2)

            # target value
            target_q = rewards + gamma * (1 - terminated) * next_q  # (B,)

            # add entropy term to the target
            if self.backup_entropy:
                target_q = target_q + gamma * (1 - terminated) * alpha * (
                    -next_logprobs
                )

        # # run all critics in batch

        # run all critics in batch
        current_q = torch_vmap(self.critic_wrapper,  self.ensemble_params, self.ensemble_buffers, (obs, actions), in_dims=(0, 0, None) )  # (n_critics, B)



        loss_critic = torch_mean((current_q - target_q[None]) ** 2)
        return loss_critic

    def loss_actor(self, obs, alpha):

        print("gaussian_rlpd.py: RLPD_Gaussian.loss_actor()")

        action, logprob = self.call(
            obs,
            deterministic=False,
            reparameterize=True,
            get_logprob=True,
        )
        
        

        current_q = torch_vmap(self.critic_wrapper, self.ensemble_params, self.ensemble_buffers, (obs, action), in_dims=(0, 0, None) )  # (n_critics, B)

        current_q = torch_mean(current_q, dim=0) + alpha * (-logprob)
        loss_actor = -torch_mean(current_q)
        return loss_actor

    def loss_temperature(self, obs, alpha, target_entropy):

        print("gaussian_rlpd.py: RLPD_Gaussian.loss_temperature()")


        with torch_no_grad() as tape:
            _, logprob = self.call(
                obs,
                deterministic=False,
                get_logprob=True,
            )

        loss_alpha = -torch_mean(alpha * (logprob + target_entropy))
        return loss_alpha











    def update_target_critic(self, tau):
        """need to use ensemble_params instead of critic_networks"""
        print("gaussian_rlpd.py: RLPD_Gaussian.update_target_critic()")

        for target_ind, target_critic in enumerate(self.target_networks):
            for target_param_name, target_param in target_critic.trainable_variables:
                source_param = self.ensemble_params[target_param_name][target_ind]
                updated_value = target_param * (1.0 - tau) + source_param * tau
                target_param.assign(updated_value)













































































