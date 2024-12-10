"""
Solving probabilistic ODE for exact likelihood, from https://github.com/yang-song/score_sde_pytorch
"""


# import torch
import numpy as np

# from torchdiffeq import odeint

# adjoint can reduce memory, but not faster
# from torchdiffeq import odeint_adjoint as odeint
from model.diffusion.sde_lib import get_score_fn

from util.torch_to_tf import torch_repeat_interleave, torch_round, torch_sum, torch_prod, torch_tensor, \
torch_linspace, torch_zeros, torch_cat, torch_hstack, torch_reshape, torch_mean, torch_full, torch_randn, torch_randint, torch_tensor_view



def get_likelihood_fn(
    sde,
    hutchinson_type="Rademacher",
    rtol=1e-5,
    atol=1e-5,
    method="RK45",
    steps=10,  # should not matter, only t_eval
    step_size=1e-3,
    eps=1e-5,
    continuous=False,
    probability_flow=False,
    predict_epsilon=False,
    num_epsilon=1,
):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      inverse_scaler: The inverse data normalizer.
      hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
      rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
      atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
      method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.
      eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

    Returns:
      A function that a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """

    print("exact_likelihood.py: get_likelihood_fn()")



    def drift_fn(
        model,
        x,
        t,
        **kwargs,
    ):
        """The drift function of the reverse-time SDE."""

        print("exact_likelihood.py: drift_fn()")

        score_fn = get_score_fn(
            sde,
            model,
            continuous=continuous,
            predict_epsilon=predict_epsilon,
        )
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=probability_flow)
        sde_out = rsde.sde(x, t, **kwargs)[0]
        return sde_out



    def div_fn(
        model,
        x,
        t,
        noise,
        create_graph=False,
        **kwargs,
    ):

        print("exact_likelihood.py: div_fn()")

        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch_sum(drift_fn(model, x, t, **kwargs) * noise)
            grad_fn_eps = torch.autograd.grad(
                fn_eps,
                x,
                create_graph=create_graph,
            )[0]
        if not create_graph:
            x.requires_grad_(False)
        return torch_sum(grad_fn_eps * noise, dim=(1, 2))



    def likelihood_fn(
        model,
        model_ft,
        data,
        denoising_steps,
        ft_denoising_steps,
        cond,
        **kwargs,
    ):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
            data: (B x Ta x Da)

        Returns:
            logprob: (B,)
        """

        print("exact_likelihood.py: likelihood_fn()")

        shape = data.shape
        B, H, A = shape
        # device = data.device

        # sample epsilon
        if hutchinson_type == "Gaussian":
            epsilon = torch_randn(size=(B * num_epsilon, H, A)
                                #   , device=device
                                  )
        elif hutchinson_type == "Rademacher":
            epsilon = (
                torch_randint(
                    low=0, high=2, size=(B * num_epsilon, H, A)
                    # , device=device
                ).float()
                * 2
                - 1.0
            )
        else:
            raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

        # repeat for expectation
        cond_eps = {
            key: torch_repeat_interleave(cond[key], num_epsilon, dim=0) for key in cond
        }



        def ode_func(t, x):

            print("exact_likelihood.py: ode_func()")

            x = x[:, :-1]
            vec_t = torch_full(
                (x.shape[0],),
                torch_round(t * denoising_steps),
                # device=x.device,
                dtype=int,
            )
            if torch_round(t * denoising_steps) <= ft_denoising_steps:
                model_fn = model_ft
            else:
                model_fn = model

            # x = x.view(shape)  # B x horizon x action_dim
            x = torch_tensor_view(x, shape)

            drift = drift_fn(
                model_fn,
                x,
                vec_t,
                cond=cond,
                **kwargs,
            )
            drift = torch_reshape(drift, B, -1)

            # repeat for expectation
            x = torch_repeat_interleave(x, num_epsilon, dim=0)
            vec_t = torch_repeat_interleave(vec_t, num_epsilon)

            logp_grad = div_fn(
                model,
                x,
                vec_t,
                epsilon,
                create_graph=True,
                cond=cond_eps,
                **kwargs,
            )[:, None]

            logp_grad = torch_reshape(logp_grad, B, num_epsilon, -1)

            logp_grad = torch_mean(logp_grad, dim=1)  # expectation over epsilon
            return torch_cat(
                [drift, logp_grad], dim=-1
            )  # Concatenate along the feature dimension

        # flatten data
        # data = data.view(shape[0], -1)

        data = torch_tensor_view(data, shape[0], -1)

        init = torch_hstack(
            (data, torch_zeros((shape[0], 1)).to(data.dtype)
            #  .to(device)
            )
        )
        t_eval = torch_linspace(eps, sde.T, steps=steps)
        # .to(device)  # eval points
        solution = odeint(
            ode_func,
            init,
            t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            options={"step_size": step_size},
            # args=(model, epsilon),
        )  # steps x batch x 3
        zp = solution[-1]  # batch x 3
        z = torch_tensor_view( zp[:, :-1], shape )

        delta_logp = zp[:, -1]
        prior_logp = sde.prior_logp(z)
        N = torch_prod(torch_tensor(shape[1:]))
        # print("prior:", prior_logp / (np.log(2) * N))
        # print("delta:", delta_logp / (np.log(2) * N))
        logprob = (prior_logp + delta_logp) / (np.log(2) * N)
        return logprob

    return likelihood_fn





