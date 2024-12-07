import logging

import os

import tensorflow as tf

import pickle
import numpy as np

from agent.dataset.d3il_dataset.base_dataset import TrajectoryDataset


class Avoiding_Dataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        device="cpu",
        obs_dim: int = 20,
        action_dim: int = 2,
        max_len_data: int = 256,
        window_size: int = 1,
    ):

        print("avoiding_dataset.py: __init__()")

        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size,
        )

        logging.info("Loading Sorting Dataset")

        inputs = []
        actions = []
        masks = []

        data_dir = data_directory
        state_files = os.listdir(data_dir)

        for file in state_files:
            with open(os.path.join(data_dir, file), "rb") as f:
                env_state = pickle.load(f)

            zero_obs = np.zeros((1, self.max_len_data, self.obs_dim), dtype=np.float32)
            zero_action = np.zeros(
                (1, self.max_len_data, self.action_dim), dtype=np.float32
            )
            zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

            # robot and box posistion
            robot_des_pos = env_state["robot"]["des_c_pos"][:, :2]
            robot_c_pos = env_state["robot"]["c_pos"][:, :2]

            input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)

            vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
            valid_len = len(vel_state)

            zero_obs[0, :valid_len, :] = input_state[:-1]
            zero_action[0, :valid_len, :] = vel_state
            zero_mask[0, :valid_len] = 1

            inputs.append(zero_obs)
            actions.append(zero_action)
            masks.append(zero_mask)

        # shape: B, T, n
        # Convert to TensorFlow tensors
        self.observations = tf.convert_to_tensor(np.concatenate(inputs), dtype=tf.float32)
        self.actions = tf.convert_to_tensor(np.concatenate(actions), dtype=tf.float32)
        self.masks = tf.convert_to_tensor(np.concatenate(masks), dtype=tf.float32)

        self.num_data = len(self.observations)

        self.slices = self.get_slices()

    def get_slices(self):

        print("avoiding_dataset.py: get_slices()")

        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.window_size < 0:
                print(
                    f"Ignored short sequence #{i}: len={T}, window={self.window_size}"
                )
            else:
                slices += [
                    (i, start, start + self.window_size)
                    for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices



    def get_seq_length(self, idx):
        print("avoiding_dataset.py: get_seq_length()")
        return int(tf.reduce_sum(self.masks[idx]).numpy())





    def get_all_actions(self):
        print("avoiding_dataset.py: get_all_actions()")
        result = []
        for i in range(len(self.masks)):
            T = self.get_seq_length(i)
            result.append(self.actions[i, :T, :])
        return tf.concat(result, axis=0)



    def get_all_observations(self):
        print("avoiding_dataset.py: get_all_observations()")
        result = []
        for i in range(len(self.masks)):
            T = self.get_seq_length(i)
            result.append(self.observations[i, :T, :])
        return tf.concat(result, axis=0)



    def __len__(self):

        print("avoiding_dataset.py: __len__()")

        return len(self.slices)




    def __getitem__(self, idx):

        print("avoiding_dataset.py: __getitem__()")

        i, start, end = self.slices[idx]

        obs = self.observations[i, start:end]
        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        return obs, act, mask
