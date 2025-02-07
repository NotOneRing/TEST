import argparse
from pathlib import Path
import time
from functools import partial
import os
import pickle
import numpy as np
import jax, jax.numpy as jnp
import haiku as hk


import optax

from haiku.initializers import Constant


from relax.utils.fs import PROJECT_ROOT

from relax.utils.persistence import make_persist


from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union, Protocol, Dict


from relax.buffer import TreeBuffer
from relax.trainer.off_policy import OffPolicyTrainer

from relax.env import create_env, create_vector_env
from relax.utils.experience import Experience



Activation = Callable[[jax.Array], jax.Array]

Identity: Activation = lambda x: x

Metric = Dict[str, float]















def seeding(seed: Optional[int] = None) -> Tuple[np.random.Generator, int]:
    print("random_utils.py:", "seeding()")
    
    seed_seq = np.random.SeedSequence(seed)
    seed = seed_seq.entropy
    bit_generator = np.random.PCG64(seed)
    return np.random.Generator(bit_generator), seed









def fix_repr(cls):
    print("fax_utils.py:", "fix_repr()")
    
    """Delete haiku's auto-generated __repr__ method, in favor of dataclass's one"""
    del cls.__repr__
    postinit = getattr(cls, "__post_init__")
    def __post_init__(self):
        print("fax_utils.py:", "__post_init__()")
    
        postinit(self)
        if hk.running_init():
            print(self)
    cls.__post_init__ = __post_init__
    return cls









def is_broadcastable(src, dst):
    print("fax_utils.py:", "is_broadcastable()")
    
    try:
        return jnp.broadcast_shapes(src, dst) == dst
    except ValueError:
        return False









def random_key_from_data(data: jax.Array) -> jax.Array:
    print("fax_utils.py:", "random_key_from_data()")
    
    # Create a random key deterministically from data, like hashing
    mean = jnp.mean(data)
    std = jnp.std(data)
    seed = (mean * std).view(jnp.uint32)
    key = jax.random.key(seed)
    return key






@dataclass
@fix_repr
class DistributionalQNet2(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        input = jnp.concatenate((obs, act), axis=-1)
        output = mlp(self.hidden_sizes, 2, self.activation, self.output_activation)(input)
        value_mean = output[..., 0]
        value_std = jax.nn.softplus(output[..., 1])
        return value_mean, value_std



@dataclass
@fix_repr
class DACERPolicyNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, t: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te = hk.Linear(self.time_dim * 2)(te)
        te = self.activation(te)
        te = hk.Linear(self.time_dim)(te)
        input = jnp.concatenate((obs, act, te), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)

def mlp(hidden_sizes: Sequence[int], output_size: int, activation: Activation, output_activation: Activation, *, squeeze_output: bool = False) -> Callable[[jax.Array], jax.Array]:
    layers = []
    for hidden_size in hidden_sizes:
        layers += [hk.Linear(hidden_size), activation]
    layers += [hk.Linear(output_size), output_activation]
    if squeeze_output:
        layers.append(partial(jnp.squeeze, axis=-1))
    return hk.Sequential(layers)


def scaled_sinusoidal_encoding(t: jax.Array, *, dim: int, theta: int = 10000, batch_shape = None) -> jax.Array:
    print("blocks.py: ", "scaled_sinusoidal_encoding")
    
    assert dim % 2 == 0
    if batch_shape is not None:
        assert is_broadcastable(jnp.shape(t), batch_shape)

    scale = 1 / dim ** 0.5
    half_dim = dim // 2
    freq_seq = jnp.arange(half_dim) / half_dim
    inv_freq = theta ** -freq_seq

    emb = jnp.einsum('..., j -> ... j', t, inv_freq)
    emb = jnp.concatenate((
        jnp.sin(emb),
        jnp.cos(emb),
    ), axis=-1)
    emb *= scale

    if batch_shape is not None:
        emb = jnp.broadcast_to(emb, (*batch_shape, dim))

    return emb

class DiffusionModel(Protocol):
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        ...

@dataclass(frozen=True)
class BetaScheduleCoefficients:
    betas: jax.Array
    alphas: jax.Array
    alphas_cumprod: jax.Array
    alphas_cumprod_prev: jax.Array
    sqrt_alphas_cumprod: jax.Array
    sqrt_one_minus_alphas_cumprod: jax.Array
    log_one_minus_alphas_cumprod: jax.Array
    sqrt_recip_alphas_cumprod: jax.Array
    sqrt_recipm1_alphas_cumprod: jax.Array
    posterior_variance: jax.Array
    posterior_log_variance_clipped: jax.Array
    posterior_mean_coef1: jax.Array
    posterior_mean_coef2: jax.Array

    @staticmethod
    def from_beta(betas: np.ndarray):    
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = np.log(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        return BetaScheduleCoefficients(
            *jax.device_put((
                betas, alphas, alphas_cumprod, alphas_cumprod_prev,
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, log_one_minus_alphas_cumprod,
                sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod,
                posterior_variance, posterior_log_variance_clipped, posterior_mean_coef1, posterior_mean_coef2
            ))
        )

    @staticmethod
    def vp_beta_schedule(timesteps: int):    
        t = np.arange(1, timesteps + 1)
        T = timesteps
        b_max = 10.
        b_min = 0.1
        alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
        betas = 1 - alpha
        return betas

    @staticmethod
    def cosine_beta_schedule(timesteps: int):    
        s = 0.008
        t = np.arange(0, timesteps + 1) / timesteps
        alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = np.clip(betas, 0, 0.999)
        return betas


@dataclass(frozen=True)
class GaussianDiffusion:
    num_timesteps: int

    def beta_schedule(self):
        with jax.ensure_compile_time_eval():
            betas = BetaScheduleCoefficients.vp_beta_schedule(self.num_timesteps)
            return BetaScheduleCoefficients.from_beta(betas)

    def p_mean_variance(self, t: int, x: jax.Array, noise_pred: jax.Array):
        B = self.beta_schedule()
        x_recon = x * B.sqrt_recip_alphas_cumprod[t] - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
        x_recon = jnp.clip(x_recon, -1, 1)
        model_mean = x_recon * B.posterior_mean_coef1[t] + x * B.posterior_mean_coef2[t]
        model_log_variance = B.posterior_log_variance_clipped[t]
        return model_mean, model_log_variance



    def p_sample(self, key: jax.Array, model: DiffusionModel, shape: Tuple[int, ...]) -> jax.Array:    
        x_key, noise_key = jax.random.split(key)
        x = jax.random.normal(x_key, shape)
        noise = jax.random.normal(noise_key, (self.num_timesteps, *shape))

        def body_fn(x, input):
            t, noise = input
            noise_pred = model(t, x)
            model_mean, model_log_variance = self.p_mean_variance(t, x, noise_pred)
            x = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * noise
            return x, None

        t = jnp.arange(self.num_timesteps)[::-1]
        x, _ = jax.lax.scan(body_fn, x, (t, noise))
        return x



    def q_sample(self, t: int, x_start: jax.Array, noise: jax.Array):
        B = self.beta_schedule()
        return B.sqrt_alphas_cumprod[t] * x_start + B.sqrt_one_minus_alphas_cumprod[t] * noise



    def p_loss(self, key: jax.Array, model: DiffusionModel, t: jax.Array, x_start: jax.Array):
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]

        noise = jax.random.normal(key, x_start.shape)
        x_noisy = jax.vmap(self.q_sample)(t, x_start, noise)
        noise_pred = model(t, x_noisy)
        loss = optax.l2_loss(noise_pred, noise)
        return loss.mean()






class DACERParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    log_alpha: jax.Array


@dataclass
class DACERNet:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    target_entropy: float

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.num_timesteps)
    
    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
    
        policy_params, log_alpha = policy_params
        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)
        
        key, noise_key = jax.random.split(key)
        action = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
        action = action + jax.random.normal(noise_key, action.shape) * jnp.exp(log_alpha) * 0.15 # other envs 0.1
        return action.clip(-1, 1)


    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params, log_alpha = policy_params
        log_alpha = -jnp.inf
        policy_params = (policy_params, log_alpha)
        return self.get_action(key, policy_params, obs)
    

    def q_evaluate(
        self, key: jax.Array, q_params: hk.Params, obs: jax.Array, act: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q_mean, q_std = self.q(q_params, obs, act)
        z = jax.random.normal(key, q_mean.shape)
        z = jnp.clip(z, -3.0, 3.0)  # NOTE: Why not truncated normal?
        q_value = q_mean + q_std * z
        return q_mean, q_std, q_value


def create_dacer_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
) -> Tuple[DACERNet, DACERParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: DistributionalQNet2(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0)
        log_alpha = jnp.array(1.0, dtype=jnp.float32) 
        return DACERParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = DACERNet(q=q.apply, policy=policy.apply, num_timesteps=num_timesteps, act_dim=act_dim, target_entropy=-act_dim*0.9)
    return net, params



class DACEROptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class DACERTrainState(NamedTuple):
    params: DACERParams
    opt_state: DACEROptStates
    step: int
    mean_q1_std: float
    mean_q2_std: float
    entropy: float
    

class Algorithm:
    # NOTE: a not elegant blanket implementation of the algorithm interface
    def _implement_common_behavior(self, stateless_update, stateless_get_action, stateless_get_deterministic_action, stateless_get_value=None):
        self._update = jax.jit(stateless_update)
        self._get_action = jax.jit(stateless_get_action)
        self._get_deterministic_action = jax.jit(stateless_get_deterministic_action)
        if stateless_get_value is not None:
            self._get_value = jax.jit(stateless_get_value)

    def update(self, key: jax.Array, data: Experience) -> Metric:
        self.state, info = self._update(key, self.state, data)
        return {k: float(v) for k, v in info.items()}

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params(), obs)
        return np.asarray(action)

    def save(self, path: str) -> None:
        state = jax.device_get(self.state)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def save_policy_structure(self, root: os.PathLike, dummy_obs: jax.Array) -> None:
        root = Path(root)

        key = jax.random.key(0)
        stochastic = make_persist(self._get_action._fun)(key, self.get_policy_params(), dummy_obs)
        deterministic = make_persist(self._get_deterministic_action._fun)(self.get_policy_params(), dummy_obs)

        stochastic.save(root / "stochastic.pkl")
        stochastic.save_info(root / "stochastic.txt")
        deterministic.save(root / "deterministic.pkl")
        deterministic.save_info(root / "deterministic.txt")

    def get_policy_params(self):
        return self.state.params.policy

    def get_value_params(self):
        return self.state.params.value

    def warmup(self, data: Experience) -> None:
        key = jax.random.key(0)
        obs = data.obs[0]
        policy_params = self.get_policy_params()
        self._update(key, self.state, data)
        self._get_action(key, policy_params, obs)
        self._get_deterministic_action(policy_params, obs)


class DACER(Algorithm):
    def __init__(
        self,
        agent: DACERNet,
        params: DACERParams,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        tau: float = 0.005,
        delay_alpha_update: int = 10000,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_samples: int = 200,
    ):        
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.optim = optax.adam(lr)
        self.alpha_optim = optax.adam(alpha_lr)
        self.entropy = 0.0

        self.state = DACERTrainState(
            params=params,
            opt_state=DACEROptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            mean_q1_std=jnp.float32(-1.0),
            mean_q2_std=jnp.float32(-1.0),
            entropy=jnp.float32(0.0),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: DACERTrainState, data: Experience
        ) -> Tuple[DACERTrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            step, mean_q1_std, mean_q2_std = state.step, state.mean_q1_std, state.mean_q2_std
            next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key, log_alpha_key = jax.random.split(key, 5)

            reward *= self.reward_scale

            # compute target q
            next_action = self.agent.get_action(next_eval_key, (policy_params, log_alpha), next_obs)
            next_q1_mean, _, next_q1_sample = self.agent.q_evaluate(new_q1_eval_key, target_q1_params, next_obs, next_action)
            next_q2_mean, _, next_q2_sample = self.agent.q_evaluate(new_q2_eval_key, target_q2_params, next_obs, next_action)
            next_q_mean = jnp.minimum(next_q1_mean, next_q2_mean)
            next_q_sample = jnp.where(next_q1_mean < next_q2_mean, next_q1_sample, next_q2_sample)
            q_target = next_q_mean 
            q_target_sample = next_q_sample 
            q_backup = reward + (1 - done) * self.gamma * q_target
            q_backup_sample = reward + (1 - done) * self.gamma * q_target_sample

            # update q
            def q_loss_fn(q_params: hk.Params, mean_q_std: float) -> jax.Array:
                q_mean, q_std = self.agent.q(q_params, obs, action)
                new_mean_q_std = jnp.mean(q_std)
                mean_q_std = jax.lax.stop_gradient(
                    (mean_q_std == -1.0) * new_mean_q_std +
                    (mean_q_std != -1.0) * (self.tau * new_mean_q_std + (1 - self.tau) * mean_q_std)
                )
                q_backup_bounded = jax.lax.stop_gradient(q_mean + jnp.clip(q_backup_sample - q_mean, -3 * mean_q_std, 3 * mean_q_std))
                q_std_detach = jax.lax.stop_gradient(jnp.maximum(q_std, 0))
                epsilon = 0.1
                q_loss = -(mean_q_std ** 2 + epsilon) * jnp.mean(
                    q_mean * jax.lax.stop_gradient(q_backup - q_mean) / (q_std_detach ** 2 + epsilon) +
                    q_std * ((jax.lax.stop_gradient(q_mean) - q_backup_bounded) ** 2 - q_std_detach ** 2) / (q_std_detach ** 3 + epsilon)
                )
                return q_loss, (q_mean, q_std, mean_q_std)

            (q1_loss, (q1_mean, q1_std, mean_q1_std)), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params, mean_q1_std)
            (q2_loss, (q2_mean, q2_std, mean_q2_std)), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params, mean_q2_std)
            
            def cal_entropy():
                keys = jax.random.split(log_alpha_key, self.num_samples)
                actions = jax.vmap(self.agent.get_action, in_axes=(0, None, None), out_axes=1)(keys, (policy_params, jax.lax.stop_gradient(log_alpha)), obs)
                entropy = jax.pure_callback(estimate_entropy, jax.ShapeDtypeStruct((), jnp.float32), actions)
                entropy = jax.lax.stop_gradient(entropy)
                return entropy
            
            prev_entropy = state.entropy if hasattr(state, 'entropy') else jnp.float32(0.0)
            
            entropy = jax.lax.cond(
                step % self.delay_alpha_update == 0,
                cal_entropy,
                lambda: prev_entropy
            )
            
            # update policy
            def policy_loss_fn(policy_params) -> jax.Array:
                new_action = self.agent.get_action(new_eval_key, (policy_params, log_alpha), obs)
                q1_mean, _ = self.agent.q(q1_params, obs, new_action)
                q2_mean, _ = self.agent.q(q2_params, obs, new_action)
                q_mean = jnp.minimum(q1_mean, q2_mean)
                policy_loss = jnp.mean(-q_mean) 
                return policy_loss

            total_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)
            
            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                log_alpha_loss = -jnp.mean(log_alpha * (-entropy + self.agent.target_entropy))
                return log_alpha_loss

            # update networks
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )
                
            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda params, opt_state: param_update(optim, params, jax.grad(log_alpha_loss_fn)(params), opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )
                
            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.optim, policy_params, policy_grads, policy_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)

            state = DACERTrainState(
                params=DACERParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha),
                opt_state=DACEROptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state),
                step=step + 1,
                mean_q1_std=mean_q1_std,
                mean_q2_std=mean_q2_std,
                entropy=entropy,
            )
            info = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1_mean),
                "q1_std": jnp.mean(q1_std),
                "q2_loss": q2_loss,
                "q2_mean": jnp.mean(q2_mean),
                "q2_std": jnp.mean(q2_std),
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                "mean_q1_std": mean_q1_std,
                "mean_q2_std": mean_q2_std,
                "entropy": entropy,
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha)


def estimate_entropy(actions, num_components=3):  # (batch, sample, dim)
    import numpy as np
    from sklearn.mixture import GaussianMixture
    total_entropy = []
    for action in actions:
        gmm = GaussianMixture(n_components=num_components, covariance_type='full')
        gmm.fit(action)
        weights = gmm.weights_
        entropies = []
        for i in range(gmm.n_components):
            cov_matrix = gmm.covariances_[i]
            d = cov_matrix.shape[0]
            entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(cov_matrix)[1]
            entropies.append(entropy)
        entropy = -np.sum(weights * np.log(weights)) + np.sum(weights * np.array(entropies))
        total_entropy.append(entropy)
    final_entropy = sum(total_entropy) / len(total_entropy)
    return final_entropy









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="dacer")
    parser.add_argument("--env", type=str, default="Humanoid-v3")
    parser.add_argument("--num_vec_envs", type=int, default=20)
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(2e5)) # other envs 3e4
    parser.add_argument("--total_step", type=int, default=int(3e7))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)

    print("init_network_key = ", init_network_key)
    print("train_key = ", train_key)


    del init_network_seed, train_seed

    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    eval_env = None

    print("type(env) = ", type(env))
    print("type(obs_dim) = ", type(obs_dim))
    print("type(act_dim) = ", type(act_dim))

    print("env = ", env)
    print("obs_dim = ", obs_dim)
    print("act_dim = ", act_dim)


    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

    gelu = partial(jax.nn.gelu, approximate=False)

    def mish(x: jax.Array):
        return x * jnp.tanh(jax.nn.softplus(x))

    agent, params = create_dacer_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish, num_timesteps=args.diffusion_steps)

    algorithm = DACER(agent, params, lr=args.lr)

    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=1,
        evaluate_env=eval_env,
        save_policy_every=300000,
        warmup_with="random",
        log_path=PROJECT_ROOT / "logs" / args.env /
                 (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}'),
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))
    trainer.run(train_key)





































