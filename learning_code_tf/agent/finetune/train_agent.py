"""
Parent fine-tuning agent class.

"""

import os
import numpy as np
from omegaconf import OmegaConf

import tensorflow as tf
import hydra
import logging
import wandb
import random

log = logging.getLogger(__name__)
from env.gym_utils import make_async


from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, OUTPUT_POSITIONS, OUTPUT_FUNCTION_HEADER



class TrainAgent:

    def __init__(self, cfg):
        if OUTPUT_FUNCTION_HEADER:
            print("train_agent.py: TrainAgent.__init__()")

        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        # torch.manual_seed(self.seed)
        tf.random.set_seed(self.seed)

        # # Wandb
        # self.use_wandb = cfg.wandb is not None
        # if cfg.wandb is not None:
        #     wandb.init(
        #         entity=cfg.wandb.entity,
        #         project=cfg.wandb.project,
        #         name=cfg.wandb.run,
        #         config=OmegaConf.to_container(cfg, resolve=True),
        #     )

        if OUTPUT_POSITIONS:
            print("before cgf.env")

        # Make vectorized env
        self.env_name = cfg.env.name
        env_type = cfg.env.get("env_type", None)

        if OUTPUT_POSITIONS:
            print("after cgf.env")
        
        if OUTPUT_VARIABLES:
            print("make_async parameters: cfg.env.name = ", cfg.env.name)
            print("make_async parameters: env_type = ", env_type)
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

        if OUTPUT_POSITIONS:
            print("after make_async")


        if not env_type == "furniture":
            self.venv.seed(
                [self.seed + i for i in range(cfg.env.n_envs)]
            )  # otherwise parallel envs might have the same initial states!
            # isaacgym environments do not need seeding


        if OUTPUT_POSITIONS:
            print("train_agent.py: TrainAgent.__init__(): 1")

        self.n_envs = cfg.env.n_envs
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = cfg.env.max_episode_steps
        self.reset_at_iteration = cfg.env.get("reset_at_iteration", True)
        self.save_full_observations = cfg.env.get("save_full_observations", False)

        if OUTPUT_POSITIONS:
            print("train_agent.py: TrainAgent.__init__(): 2")

        self.furniture_sparse_reward = (
            cfg.env.specific.get("sparse_reward", False)
            if "specific" in cfg.env
            else False
        )  # furniture specific, for best reward calculation

        if OUTPUT_POSITIONS:
            print("train_agent.py: TrainAgent.__init__(): 3")

        # Batch size for gradient update
        self.batch_size: int = cfg.train.batch_size


        if OUTPUT_POSITIONS:
            print("train_agent.py: TrainAgent.__init__(): 4")


        if OUTPUT_VARIABLES:
            print("cfg.model = ", cfg.model)

        self.cfg_env_name = cfg.env_name


        if OUTPUT_VARIABLES:
            print("self.cfg_env_name = ", self.cfg_env_name)        

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model, env_name=self.cfg_env_name)





        if OUTPUT_POSITIONS:
            print("train_agent.py: TrainAgent.__init__(): 5")

        # Training params
        self.itr = 0
        self.n_train_itr = cfg.train.n_train_itr
        self.val_freq = cfg.train.val_freq
        self.force_train = cfg.train.get("force_train", False)
        self.n_steps = cfg.train.n_steps

        if OUTPUT_POSITIONS:
            print("train_agent.py: TrainAgent.__init__(): 6")

        self.best_reward_threshold_for_success = (
            len(self.venv.pairs_to_assemble)
            if env_type == "furniture"
            else cfg.env.best_reward_threshold_for_success
        )
        self.max_grad_norm = cfg.train.get("max_grad_norm", None)


        if OUTPUT_POSITIONS:
            print("train_agent.py: TrainAgent.__init__(): 7")

        # Logging, rendering, checkpoints
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        self.result_path = os.path.join(self.logdir, "result.pkl")
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_trajs = cfg.train.get("save_trajs", False)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq
        self.render_freq = cfg.train.render.freq
        self.n_render = cfg.train.render.num
        self.render_video = cfg.env.get("save_video", False)

        if OUTPUT_POSITIONS:
            print("train_agent.py: TrainAgent.__init__(): 8")

        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"

        if OUTPUT_POSITIONS:
            print("train_agent.py: TrainAgent.__init__(): 9")

        assert not (
            self.n_render <= 0 and self.render_video
        ), "Need to set n_render > 0 if saving video"
        

        if OUTPUT_POSITIONS:
            print("train_agent.py: TrainAgent.__init__(): 10")


        self.traj_plotter = (
            hydra.utils.instantiate(cfg.train.plotter)
            if "plotter" in cfg.train
            else None
        )


        if OUTPUT_POSITIONS:
            print("train_agent.py: TrainAgent.__init__(): 11")


        

    def run(self):
        print("train_agent.py: TrainAgent.run()")
        pass


    def save_model(self, learn_eta):
        """
        Saves model to disk.
        """

        if OUTPUT_FUNCTION_HEADER:
            print("train_agent.py: TrainAgent.save_model()")

        savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.keras")

        print("save_model savepath = ", savepath)

        print("finetune: train_agent.save_model: savepath = ", savepath)
        log.info(f"Saved model to {savepath}")


        tf.keras.models.save_model(self.model, savepath)
        print(f"Saved model to {savepath}")


        actor_ft_savepath = savepath.replace(".keras", "_actor_ft.keras")
        print("actor_ft_savepath = ", actor_ft_savepath)
        tf.keras.models.save_model(self.model.actor_ft, actor_ft_savepath)
        print(f"Saved model.actor_ft to {actor_ft_savepath}")


        critic_savepath = savepath.replace(".keras", "_critic.keras")
        print("critic_savepath = ", critic_savepath)
        tf.keras.models.save_model(self.model.critic, critic_savepath)
        print(f"Saved model.critic to {critic_savepath}")

        if learn_eta:
            eta_savepath = savepath.replace(".keras", "_eta.keras")
            print("eta_savepath = ", eta_savepath)
            tf.keras.models.save_model(self.model.eta, eta_savepath)
            print(f"Saved model.eta to {eta_savepath}")



    # def save_model(self):
    #     """
    #     saves model to disk; no ema
    #     """

    #     print("train_agent.py: TrainAgent.save_model()")

    #     data = {
    #         "itr": self.itr,
    #         "model": self.model.state_dict(),
    #     }
    #     savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
    #     torch.save(data, savepath)
    #     log.info(f"Saved model to {savepath}")


    def load(self, itr):
        """
        Loads model from disk.
        """

        if OUTPUT_FUNCTION_HEADER:
            print("train_agent.py: TrainAgent.load()")

        loadpath = os.path.join(self.checkpoint_dir, f"state_{itr}.keras")

        if OUTPUT_VARIABLES:
            print("loadpath = ", loadpath)

        from model.diffusion.mlp_diffusion import DiffusionMLP
        from model.diffusion.diffusion import DiffusionModel
        from model.common.mlp import MLP, ResidualMLP, TwoLayerPreActivationResNetLinear
        from model.diffusion.modules import SinusoidalPosEmb
        from model.common.modules import SpatialEmb, RandomShiftsAug
        from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish, nn_Identity

        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
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


        self.model = tf.keras.models.load_model(loadpath,  custom_objects=get_custom_objects() )


        self.model.actor_ft = tf.keras.models.load_model(loadpath.replace(".keras", "_actor_ft.keras") ,  custom_objects=get_custom_objects() )
        self.model.critic = tf.keras.models.load_model(loadpath.replace(".keras", "_critic.keras") ,  custom_objects=get_custom_objects() )
        self.model.eta = tf.keras.models.load_model(loadpath.replace(".keras", "_eta.keras") ,  custom_objects=get_custom_objects() )


    # def load(self, itr):
    #     """
    #     loads model from disk
    #     """

    #     print("train_agent.py: TrainAgent.load()")

    #     loadpath = os.path.join(self.checkpoint_dir, f"state_{itr}.pt")
    #     data = torch.load(loadpath, weights_only=True)

    #     self.itr = data["itr"]
    #     self.model.load_state_dict(data["model"])

    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        
        if OUTPUT_FUNCTION_HEADER:
            print("train_agent.py: TrainAgent.reset_env_all()")

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

        if OUTPUT_FUNCTION_HEADER:
            print("train_agent.py: TrainAgent.reset_env()")

        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs
