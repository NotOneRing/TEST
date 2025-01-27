"""
Parent eval agent class.

"""

import os
import numpy as np
# import torch
import tensorflow as tf
import hydra
import logging
import random

log = logging.getLogger(__name__)
from env.gym_utils import make_async

from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, OUTPUT_POSITIONS, OUTPUT_FUNCTION_HEADER


class EvalAgent:

    def __init__(self, cfg):

        print("eval_agent.py: EvalAgent.__init__()")

        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)


        print("cfg = ", cfg)


        # Make vectorized env
        self.env_name = cfg.env.name

        if OUTPUT_VARIABLES:
            print("self.env_name = ", self.env_name)


        if OUTPUT_VARIABLES:
            print("make_async parameters: cfg.env.name = ", cfg.env.name)
            print("make_async parameters: env_type = ", cfg.env.get("env_type", None))
            print("make_async parameters: cfg.env.n_envs = ", cfg.env.n_envs)
            print("make_async parameters: cfg.env.max_episode_steps = ", cfg.env.max_episode_steps)
            print("make_async parameters: wrappers = ", cfg.env.get("wrappers", None))
            print("make_async parameters: robomimic_env_cfg_path = ", cfg.get("robomimic_env_cfg_path", None))
            print("make_async parameters: shape_meta = ", cfg.get("shape_meta", None))
            print("make_async parameters: use_image_obs = ", cfg.env.get("use_image_obs", False))
            print("make_async parameters: render = ", cfg.env.get("render", False))
            print("make_async parameters: render_offscreen = ", cfg.env.get("save_video", False))
            print("make_async parameters: obs_dim = ", cfg.obs_dim)
            print("make_async parameters: action_dim = ", cfg.action_dim)


        env_type = cfg.env.get("env_type", None)
        self.venv = make_async(
            cfg.env.name,
            env_type=env_type,
            num_envs=cfg.env.n_envs,
            asynchronous=True,
            max_episode_steps=cfg.env.max_episode_steps,
            wrappers=cfg.env.get("wrappers", None),
            robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
            shape_meta=cfg.get("shape_meta", None),
            use_image_obs=cfg.env.get("use_image_obs", False),
            render=cfg.env.get("render", False),
            render_offscreen=cfg.env.get("save_video", False),
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            **cfg.env.specific if "specific" in cfg.env else {},
        )
        if not env_type == "furniture":
            self.venv.seed(
                [self.seed + i for i in range(cfg.env.n_envs)]
            )  # otherwise parallel envs might have the same initial states!
            # isaacgym environments do not need seeding
        self.n_envs = cfg.env.n_envs
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = cfg.env.max_episode_steps
        self.reset_at_iteration = cfg.env.get("reset_at_iteration", True)
        self.save_full_observations = cfg.env.get("save_full_observations", False)
        self.furniture_sparse_reward = (
            cfg.env.specific.get("sparse_reward", False)
            if "specific" in cfg.env
            else False
        )  # furniture specific, for best reward calculation

        if OUTPUT_VARIABLES:
            print("cfg.model = ", cfg.model)

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model, env_name=self.env_name)

        # Eval params
        self.n_steps = cfg.n_steps
        self.best_reward_threshold_for_success = (
            len(self.venv.pairs_to_assemble)
            if env_type == "furniture"
            else cfg.env.best_reward_threshold_for_success
        )

        # Logging, rendering
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.render_dir, exist_ok=True)
        self.n_render = cfg.render_num
        self.render_video = cfg.env.get("save_video", False)
        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        assert not (
            self.n_render <= 0 and self.render_video
        ), "Need to set n_render > 0 if saving video"
        self.traj_plotter = (
            hydra.utils.instantiate(cfg.plotter)
            if "plotter" in cfg else None
        )


    def run(self):
        print("eval_agent.py: EvalAgent.run()")
        pass

    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        
        print("eval_agent.py: EvalAgent.reset_env_all()")

        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        # convert to OrderedDict if obs_venv is a list of dict
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        if verbose:
            for index in range(self.n_envs):
                logging.info(
                    f"<-- Reset environment {index} with options {options_venv[index]}"
                )
        return obs_venv

    def reset_env(self, env_ind, verbose=False):

        print("eval_agent.py: EvalAgent.reset_env()")

        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs



















