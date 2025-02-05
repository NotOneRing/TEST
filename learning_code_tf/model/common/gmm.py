"""
GMM policy parameterization.

"""

import tensorflow as tf
# import tensorflow_probability as tfp

import logging

log = logging.getLogger(__name__)

from util.torch_to_tf import Normal, Categorical, Independent, MixtureSameFamily, \
torch_tensor_view, torch_mean, torch_mean, torch_sum, torch_softmax, torch_ones_like



class GMMModel(tf.keras.Model):

    def __init__(
        self,
        network,
        horizon_steps,
        network_path=None,
        device="cuda:0",
        **kwargs,
    ):

        print("gmm.py: GMMModel.__init__()")

        super().__init__()
        self.network = network
        if network_path is not None:
            print("self = ", self)
            print("self.network = ", self.network)
            print("checkpoint = ", checkpoint)
            
            checkpoint = tf.train.Checkpoint(model=self)
            checkpoint.restore(network_path).expect_partial()
            logging.info("Loaded actor from %s", network_path)
            

        log.info(
            f"Number of network parameters: {sum(var.numpy().size for var in self.trainable_variables)}"
        )

        self.horizon_steps = horizon_steps

    def loss_ori(
        self,
        true_action,
        cond,
        **kwargs,
    ):

        print("gmm.py: GMMModel.loss()")

        B = tf.shape(true_action)[0]

        dist, entropy, _ = self.forward_train(
            cond,
            deterministic=False,
        )
        true_action = torch_tensor_view(true_action, [B, -1])
        loss = -dist.log_prob(true_action)  # [B]
        loss = torch_mean(loss)
        return loss, {"entropy": entropy}



    def forward_train(
        self,
        cond,
        deterministic=False,
    ):
        """
        Calls the MLP to compute the mean, scale, and logits of the GMM. Returns the torch.Distribution object.
        """

        print("gmm.py: GMMModel.forward_train()")

        means, scales, logits = self.network(cond)
        if deterministic:
            # low-noise for all Gaussian dists
            scales = torch_ones_like(means) * 1e-4

        # mixture components - make sure that `batch_shape` for the distribution is equal to (batch_size, num_modes) since MixtureSameFamily expects this shape
        # Each mode has mean vector of dim T*D

        # component_distribution = tfp.distributions.Normal(loc=means, scale=scales)
        component_distribution = Normal(means, scales)

        component_distribution = Independent(component_distribution, 1)

        component_entropy = component_distribution.entropy()



        approx_entropy = torch_mean(
            torch_sum(torch_softmax(logits, dim=-1) * component_entropy, dim=-1)
        )
        
        std = torch_mean(torch_sum(torch_softmax(logits, dim=-1) * torch_mean(scales, dim=-1), dim=-1))

        # Unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = Categorical(logits=logits)
        
        dist = MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        
        
        return dist, approx_entropy, std
    


    def call(self, cond, deterministic=False):

        print("gmm.py: GMMModel.call()")

        B = tf.shape(cond["state"])[0] if "state" in cond else tf.shape(cond["rgb"])[0]

        T = self.horizon_steps
        dist, _, _ = self.forward_train(
            cond,
            deterministic=deterministic,
        )

        sampled_action = dist.sample()

        sampled_action = torch_tensor_view(sampled_action, [B, T, -1])
        return sampled_action
    




















