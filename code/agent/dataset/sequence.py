"""
Pre-training data loader. Modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/sequence.py

No normalization is applied here --- we always normalize the data when pre-processing it with a different script, and the normalization info is also used in RL fine-tuning.

"""

from collections import namedtuple
import numpy as np

import tensorflow as tf

import logging
import pickle
import random
from tqdm import tqdm

log = logging.getLogger(__name__)


from util.torch_to_tf import torch_stack


class StitchedSequenceDataset:
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.

    Use the first max_n_episodes episodes (instead of random sampling)

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs (determined based on traj_lengths):  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]

    Each sample is a namedtuple of (1) chunked actions and (2) a list (obs timesteps) of dictionary with keys states and images.
    """

    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        device="GPU",
    ):
        print("sequence.py: StitchedSequenceDataset.__init__()")

        assert (
            img_cond_steps <= cond_steps
        ), "consider using more cond_steps than img_cond_steps"
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path

        # Load dataset to device specified
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)  # only np arrays
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)

        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps)

        # Extract states and actions up to max_n_episodes
        self.states = tf.convert_to_tensor(dataset["states"][:total_num_steps], dtype=tf.float32)  # (total_num_steps, obs_dim)
        self.actions = tf.convert_to_tensor(dataset["actions"][:total_num_steps], dtype=tf.float32)  # (total_num_steps, action_dim)
        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")

        if self.use_img:
            self.images = tf.convert_to_tensor(dataset["images"][:total_num_steps], dtype=tf.float32)  # (total_num_steps, C, H, W)
            log.info(f"Images shape/type: {self.images.shape, self.images.dtype}")




    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        
        # print("sequence.py: StitchedSequenceDataset.__getitem__()")
        start, num_before_start = self.indices[idx]

        # Calculate the end index
        end = start + self.horizon_steps
        
        states = self.states[(start - num_before_start) : (start + 1), :]

        actions = self.actions[start:end, :]
        
        # Stack the states in reverse order of time, so that the most recent state is at the last
        states = tf.stack(
            [
                # states[max(num_before_start - t, 0)]
                states[max(num_before_start - t, 0), :]
                for t in reversed(range(self.cond_steps))
            ]
        ) 
        
        returned_dict = {"states": states}


        # If images are used, adjust the start index by subtracting the steps prior to the start
        if self.use_img:
            
            images = self.images[(start - num_before_start) : end]

            # Stack the images in reverse order of time, so that the most recent image is at the last
            images = torch_stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )

            # Add images to the condition dictionary
            returned_dict['rgb'] = images

        # Create a batch of actions and conditions
        returned_dict['actions'] = actions

        return returned_dict



    def make_indices(self, traj_lengths, horizon_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory
        """

        print("sequence.py: StitchedSequenceDataset.make_indices()")

        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        return indices


    def __len__(self):
        return len(self.indices)


    def set_train_val_split(self, train_split):
        """
        Not doing validation right now
        """

        print("sequence.py: StitchedSequenceDataset.set_train_val_split()")

        num_train = int(len(self.indices) * train_split)
        train_indices = random.sample(self.indices, num_train)
        val_indices = [i for i in range(len(self.indices)) if i not in train_indices]
        self.indices = train_indices
        return val_indices


    def _inputs(self):
        return None




class StitchedSequenceQLearningDataset:
    def __init__(self, dataset_path, horizon_steps=64, cond_steps=1, img_cond_steps=1, max_n_episodes=10000, use_img=False, device="/GPU:0"):
        print("sequence.py: StitchedSequenceQLearningDataset.__init__()")

        assert img_cond_steps <= cond_steps, "consider using more cond_steps than img_cond_steps"
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path

        # Load dataset
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)

        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps)


        print("type(dataset) = ", type(dataset))
        print("dataset = ", dataset)



        # Extract states, actions, rewards, and next_states
        self.states = tf.convert_to_tensor(dataset["states"][:total_num_steps], dtype=tf.float32)  # (total_num_steps, obs_dim)
        self.actions = tf.convert_to_tensor(dataset["actions"][:total_num_steps], dtype=tf.float32)  # (total_num_steps, action_dim)
        self.rewards = tf.convert_to_tensor(dataset["rewards"][:total_num_steps], dtype=tf.float32)  # (total_num_steps,)
        self.next_states = tf.convert_to_tensor(dataset["next_states"][:total_num_steps], dtype=tf.float32)  # (total_num_steps, obs_dim)
        
        print(f"States shape: {self.states.shape}")
        print(f"Actions shape: {self.actions.shape}")
        print(f"Rewards shape: {self.rewards.shape}")
        print(f"Next states shape: {self.next_states.shape}")

        if self.use_img:
            self.images = tf.convert_to_tensor(dataset["images"][:total_num_steps], dtype=tf.float32)  # (total_num_steps, C, H, W)
            print(f"Images shape: {self.images.shape}")

    def make_indices(self, traj_lengths, horizon_steps):
        """
        Makes indices for sampling from dataset;
        each index maps to a datapoint, also saves the number of steps before it within the same trajectory.
        """
        print("sequence.py: StitchedSequenceQLearningDataset.make_indices()")

        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [(i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)]
            cur_traj_index += traj_length
        return indices

    def __getitem__(self, idx):
        """
        Fetch the data for the given index (idx), including actions, rewards, next states, and conditions.
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start):(start + 1)]
        actions = self.actions[start:end]
        rewards = self.rewards[start:end]
        next_states = self.next_states[start:end]
        
        # Stack states and images (if used)
        states = tf.stack([states[max(num_before_start - t, 0)] for t in reversed(range(self.cond_steps))])
        
        conditions = {"state": states}
        if self.use_img:

            images = self.images[(start - num_before_start):end]
            images = torch_stack([images[max(num_before_start - t, 0)] for t in reversed(range(self.img_cond_steps))])
            conditions["rgb"] = images
            returned_dict["rgb"] = images


        returned_dict = {"actions": actions}

        returned_dict = {"states": states}

        returned_dict["rewards"] = rewards
        returned_dict["next_states"] = next_states
        returned_dict['conditions'] = conditions

        return returned_dict


    def __len__(self):
        return len(self.indices)




