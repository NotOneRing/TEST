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

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones")
TransitionWithReturn = namedtuple(
    "Transition", "actions conditions rewards dones reward_to_gos"
)

class StitchedSequenceDataset(tf.data.Dataset):
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




    # def __iter__(self):
    #     # for idx in range(len(self)):
    #     #     yield self[idx]

    #     for idx in range(len(self)):
    #         sample = self[idx]
    #         assert sample[0].shape == (self.horizon_steps, self.actions.shape[-1]), \
    #             f"Action shape mismatch: {sample[0].shape} != {(self.horizon_steps, self.actions.shape[-1])}"
    #         assert sample[1]["state"].shape == (self.cond_steps, self.states.shape[-1]), \
    #             f"State shape mismatch: {sample[1]['state'].shape} != {(self.cond_steps, self.states.shape[-1])}"
    #         yield sample


    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        
        # print("sequence.py: StitchedSequenceDataset.__getitem__()")

        # 从索引列表中获取开始索引和开始前的步数
        start, num_before_start = self.indices[idx]

        # 计算结束索引
        end = start + self.horizon_steps

        # print("Start index:", start)
        # print("Num before start:", num_before_start)
        print("End index:", end)


        # print("start = ", start)
        # print("end = ", end)
        # print("start - num_before_start = ", start - num_before_start)
        # print("(start + 1) = ", (start + 1))
        # print("type(self.states) = ", type(self.states))
        # print("len(self.states) = ", len(self.states))
        # print("self.states.shape = ", self.states.shape)



        # # 从开始索引减去开始前的步数到开始索引加1获取状态
        # states = self.states[(start - num_before_start) : (start + 1)]
        states = self.states[(start - num_before_start) : (start + 1), :]

        # print("type(states) = ", type(states))
        # print("states.shape = ", states.shape)

        # print("start = ", start)
        # print("end = ", end)
        # print("type(self.actions) = ", type(self.actions))

        # print("len(self.actions) = ", len(self.actions))
        # print("self.actions.shape = ", self.actions.shape)

        # 从开始索引到结束索引获取动作
        # actions = self.actions[start:end]


        actions = self.actions[start:end, :]

        # print("type(actions) = ", type(actions))
        # print("actions.shape = ", actions.shape)

        # print("self.cond_steps = ", self.cond_steps)


        # 将状态按时间倒序堆叠，因此最近的状态在最后
        states = tf.stack(
            [
                # states[max(num_before_start - t, 0)]
                states[max(num_before_start - t, 0), :]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end


        # print("stack: type(states) = ", type(states))
        # print("stack: states.shape = ", states.shape)


        # # 创建一个字典来存储条件
        # conditions = {"states": states}

        # print("before returned dict")
        returned_dict = {"states": states}

        # print("after returned dict")

        # 如果使用图像，则从开始索引减去开始前的步数到结束索引获取图像
        if self.use_img:
            raise NotImplementedError("use_img: dimension check is not implemented now.")
            
            images = self.images[(start - num_before_start) : end]

            # 将图像按时间倒序堆叠，因此最近的图像在最后
            images = tf.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )

            # 将图像添加到条件字典中
            conditions["rgb"] = images
            returned_dict['rgb'] = images

        # # 创建一个包含动作和条件的批次
        # batch = Batch(actions, conditions)
        returned_dict['actions'] = actions

        # print("type(batch) = ", type(batch))
        # print("batch = ", batch)

        # # 返回批次
        # return batch
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
        # 数据集没有父数据集作为输入，因此返回 None
        return None


    @property
    def element_spec(self):
        # 定义动作的 TensorSpec
        action_spec = tf.TensorSpec(
            shape=(self.horizon_steps, self.actions.shape[-1]),
            dtype=self.actions.dtype,
        )

        # 定义状态的 TensorSpec
        state_spec = tf.TensorSpec(
            shape=(self.cond_steps, self.states.shape[-1]),
            dtype=self.states.dtype,
        )

        # 定义 conditions 的字典结构
        spec = {"state": state_spec}
        if self.use_img:
            img_spec = tf.TensorSpec(
                shape=(self.img_cond_steps,) + self.images.shape[1:],
                dtype=self.images.dtype,
            )
            spec["rgb"] = img_spec
        # 返回一个 Batch 的 TensorSpec
        return (action_spec, spec)





class StitchedSequenceQLearningDataset(tf.data.Dataset):
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
            images = tf.stack([images[max(num_before_start - t, 0)] for t in reversed(range(self.img_cond_steps))])
            conditions["rgb"] = images
        
        batch = (actions, rewards, next_states, conditions)
        return batch

    def __len__(self):
        return len(self.indices)




