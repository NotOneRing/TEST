"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

From https://github.com/yang-song/score_sde_pytorch

"""

import abc

import numpy as np


from util.torch_to_tf import torch_sum, torch_sqrt, torch_tensor, torch_randn, torch_cumprod, torch_zeros_like

from util.torch_to_tf import torch_tensor_long, torch_clamp


def get_score_fn(
    sde,
    model,
    continuous=False,
    predict_epsilon=False,
):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score model.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
      A score function.
    """

    print("sde_lib.py: get_score_fn()")

    def score_fn(x, t, **kwargs):
        """
        Use [:, None, None] to add two dimensions (horizon and transition)
        """

        print("sde_lib.py: score_fn()")
        
        score = model(x, t, **kwargs)

        if not predict_epsilon:  # get epsilon first from predicted mu
            score = (
                -(x - score * sde.sqrt_alphas[ torch_tensor_long(t) ][:, None, None])
                / sde.discrete_betas[ torch_tensor_long(t) ][:, None, None]
            )
        else:
            std = sde.sqrt_1m_alpha_bar[ torch_tensor_long(t) ]
            score = -score / std[:, None, None]
        return score

    return score_fn


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.
        Args:
          N: number of discretization time steps.
        """

        print("sde_lib.py: SDE.__init__()")

        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""

        print("sde_lib.py: SDE.T()")

        pass

    @abc.abstractmethod
    def sde(self, x, t):

        print("sde_lib.py: SDE.sde()")

        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""

        print("sde_lib.py: SDE.marginal_prob()")

        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""

        print("sde_lib.py: SDE.prior_sampling()")

        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """

        print("sde_lib.py: SDE.prior_logp()")

        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """

        print("sde_lib.py: SDE.discretize()")

        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch_sqrt( torch_tensor(dt) )
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """

        print("sde_lib.py: SDE.reverse()")

        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):

                print("sde_lib.py: RSDE.__init__()")

                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):

                print("sde_lib.py: RSDE.T()")

                return T

            def sde(self, x, t, **kwargs):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""

                print("sde_lib.py: RSDE.sde()")

                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t, **kwargs)
                drift = drift - diffusion[:, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""

                print("sde_lib.py: RSDE.discretize()")

                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None] ** 2 * score_fn(x, t) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch_zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    def __init__(self, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """

        print("sde_lib.py: VPSDE.__init__()")

        super().__init__(N)

    def set_betas(self, betas, min_beta=0.01):

        print("sde_lib.py: VPSDE.set_betas()")

        self.discrete_betas = torch_clamp(betas, min=min_beta)  # cosine schedule from our DDPM
        self.alphas = 1.0 - self.discrete_betas
        self.sqrt_alphas = torch_sqrt(self.alphas)
        self.alphas_bar = torch_cumprod(self.alphas, axis=0)
        self.sqrt_1m_alpha_bar = torch_sqrt(1 - self.alphas_bar)

    @property
    def T(self):

        print("sde_lib.py: VPSDE.T()")

        return 1

    def sde(self, x, t):

        print("sde_lib.py: VPSDE.sde()")

        # dx = - 1/2 beta(t) x dt + sqrt(beta(t)) dW
        beta_t = self.discrete_betas[t]
        drift = -0.5 * beta_t[:, None, None] * x
        diffusion = torch_sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):

        print("sde_lib.py: VPSDE.marginal_prob()")

        raise NotImplementedError
        # log_mean_coeff = (
        #     -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        # )
        # mean = torch.exp(log_mean_coeff[:, None, None]) * x
        # std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        # return mean, std

    def prior_sampling(self, shape):

        print("sde_lib.py: VPSDE.prior_sampling()")

        return torch_randn(*shape)

    def prior_logp(self, z):

        print("sde_lib.py: VPSDE.prior_logp()")

        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch_sum(z**2, dim=(1, 2)) / 2.0
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""

        print("sde_lib.py: VPSDE.discretize()")

        timestep = torch_tensor_long(t * (self.N - 1) / self.T)
        beta = self.discrete_betas[timestep]
        alpha = self.alphas[timestep]

        sqrt_beta = torch_sqrt(beta)
        f = torch_sqrt(alpha)[:, None, None] * x - x
        G = sqrt_beta
        return f, G







