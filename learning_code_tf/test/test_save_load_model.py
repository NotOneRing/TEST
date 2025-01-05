"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.

"""

import os
import sys
import pretty_errors
import logging

import math
import hydra
from omegaconf import OmegaConf
import gdown
from download_url import (
    get_dataset_download_url,
    get_normalization_download_url,
    get_checkpoint_download_url,
)

import tensorflow as tf

tf.config.run_functions_eagerly(True)

tf.data.experimental.enable_debug_mode()


import logging

# # get TF logger
# log = logging.getLogger('tensorflow')
# log.setLevel(logging.DEBUG)

# # create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # create file handler which logs even debug messages
# fh = logging.FileHandler('tensorflow.log')
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# log.addHandler(fh)


import sys
import tensorflow as tf


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5,6,7"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# DPPO_WANDB_ENTITY=None

#输出环境变量work了
# export WANDB_MODE=disabled

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)



# suppress d4rl import error
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        os.getcwd(), "cfg"
    ),  # possibly overwritten by --config-path
)
def main(cfg: OmegaConf):

    print("run.py: main()")

    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    # For pre-training: download dataset if needed
    if "train_dataset_path" in cfg and not os.path.exists(cfg.train_dataset_path):
        download_url = get_dataset_download_url(cfg)
        download_target = os.path.dirname(cfg.train_dataset_path)
        log.info(f"Downloading dataset from {download_url} to {download_target}")
        gdown.download_folder(url=download_url, output=download_target)

    # For for-tuning: download normalization if needed
    if "normalization_path" in cfg and not os.path.exists(cfg.normalization_path):
        download_url = get_normalization_download_url(cfg)
        download_target = cfg.normalization_path
        dir_name = os.path.dirname(download_target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log.info(
            f"Downloading normalization statistics from {download_url} to {download_target}"
        )
        gdown.download(url=download_url, output=download_target, fuzzy=True)

    if "base_policy_path" in cfg:
        print("cfg.base_policy_path = ", cfg.base_policy_path)
    
    # For for-tuning: download checkpoint if needed
    if "base_policy_path" in cfg and not os.path.exists(cfg.base_policy_path):
        download_url = get_checkpoint_download_url(cfg)
        if download_url is None:
            raise ValueError(
                f"Unknown checkpoint path. Did you specify the correct path to the policy you trained?"
            )
        download_target = cfg.base_policy_path
        dir_name = os.path.dirname(download_target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log.info(f"Downloading checkpoint from {download_url} to {download_target}")
        gdown.download(url=download_url, output=download_target, fuzzy=True)

    # Deal with isaacgym needs to be imported before torch
    if "env" in cfg and "env_type" in cfg.env and cfg.env.env_type == "furniture":
        import furniture_bench


    import keras
    print(keras.__version__)

    import tensorflow.keras.backend as K
    K.clear_session()

    # run agent
    cls = hydra.utils.get_class(cfg._target_)

    print("cls = ", cls)

    # print("cfg = ", cfg)
    print("cfg = ", cfg)
    
    agent = cls(cfg)

    agent.run()
 

if __name__ == "__main__":
    main()
