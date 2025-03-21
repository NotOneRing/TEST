"""
Diffusion Actor-Critic with Entropy Regulator(DACER) agent training script.

Does not support image observations right now. 
"""

import os
import pickle
import numpy as np
# import torch

import tensorflow as tf

import einops
from copy import deepcopy


import logging
import wandb
from collections import deque

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_agent import TrainAgent

from util.torch_to_tf import *




class TrainDacerAgent(TrainAgent):
    def __init__(self, cfg):
        print("train_sac_agent.py: TrainSACAgent.__init__()")

        super().__init__(cfg)
                
        
        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma







        self.actor_optimizer = torch_optim_Adam(
            self.model.actor.trainable_variables,
            lr = cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.critic1_optimizer = torch_optim_Adam(
            self.model.critic.Q1.trainable_variables,
            lr = cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic2_optimizer = torch_optim_Adam(
            self.model.critic.Q2.trainable_variables,
            lr = cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )





        # Perturbation scale
        self.target_ema_rate = cfg.train.target_ema_rate

        # Reward scale
        self.scale_reward_factor = cfg.train.scale_reward_factor

        # Actor/critic update frequency - assume single env
        self.critic_update_freq = int(
            cfg.train.batch_size / cfg.train.critic_replay_ratio
        )
        self.actor_update_freq = int(
            cfg.train.batch_size / cfg.train.actor_replay_ratio
        )

        # Buffer size
        self.buffer_size = cfg.train.buffer_size

        # Eval episodes
        self.n_eval_episode = cfg.train.n_eval_episode

        # Exploration steps at the beginning - using randomly sampled action
        self.n_explore_steps = cfg.train.n_explore_steps


        # Initialize temperature parameter for entropy
        init_temperature = cfg.train.init_temperature

        print("init_temperature = ", init_temperature)
        print("type(init_temperature) = ", type(init_temperature))

        temp_result = np.log( np.array([init_temperature], dtype=np.float32) )

        print("temp_result = ", temp_result)
        print("type(temp_result) = ", type(temp_result))


        self.log_alpha = tf.Variable( temp_result , trainable = True)

        self.target_entropy = cfg.train.target_entropy

        print("self.target_entropy = ", self.target_entropy)
        
        self.log_alpha_optimizer = torch_optim_Adam(
            [self.log_alpha],
            lr=3e-4,
        )


        # Actor params
        self.use_expectile_exploration = cfg.train.use_expectile_exploration
        # Updates
        self.replay_ratio = cfg.train.replay_ratio
        self.critic_tau = cfg.train.critic_tau
        # Whether to use deterministic mode when sampling at eval
        self.eval_deterministic = cfg.train.get("eval_deterministic", False)
        # Sampling
        self.num_sample = cfg.train.eval_sample_num



    def run(self):
        import time

        print("train_sac_agent.py: TrainSACAgent.run()")

        # make a FIFO replay buffer for obs, action, and reward
        obs_buffer = deque(maxlen=self.buffer_size)
        next_obs_buffer = deque(maxlen=self.buffer_size)
        action_buffer = deque(maxlen=self.buffer_size)
        reward_buffer = deque(maxlen=self.buffer_size)
        terminated_buffer = deque(maxlen=self.buffer_size)

        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:
            print("self.itr = ", self.itr)

            if self.itr % 1000 == 0:
                print(f"Finished training iteration {self.itr} of {self.n_train_itr}")

            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            print("Different 1 changed: eval_mode = self.itr % self.val_freq == 0 and self.itr > self.n_explore_steps and not self.force_train")
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train

            print("Different 2 removed: n_steps = ( self.n_steps if not eval_mode else int(1e5) )")
            

            if eval_mode:
                training=False
            else:
                training=True

            print("Different 3 added: last_itr_eval = eval_mode")
            last_itr_eval = eval_mode



            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) at the beginning
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))

            print("Different 4 changed: if self.reset_at_iteration or eval_mode or self.itr == 0:")
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv

            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            # Collect a set of trajectories from env
            print("Different 5 removed: cnt_episode = 0")

            print("Different 6 changed: for step in range(n_steps)")
            # for step in range(n_steps):
            for step in range(self.n_steps):
                
                time1 = time.time()

                print("Different 7 changed: if self.itr < self.n_explore_steps")

                with torch_no_grad() as tape:
                    cond = {
                        "state": torch_tensor_float( torch_from_numpy(prev_obs_venv["state"]) )
                    }
                    samples = (
                        self.model(
                            cond=cond,
                            deterministic=eval_mode,
                        )
                        .numpy()
                    )  # n_env x horizon x act

                time2 = time.time()
                elapsed_time = time2 - time1
                print(f"Elapsed time: single Samples {elapsed_time:.4f} seconds")


                action_venv = samples[:, : self.act_steps]

                print("samples.shape = ", samples.shape)
                print("action_venv.shape = ", action_venv.shape)


                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                
                done_venv = terminated_venv | truncated_venv
                reward_trajs[step] = reward_venv
                firsts_trajs[step + 1] = done_venv


                print("Different 8 changed: add to buffer")

                if not eval_mode:
                    obs_venv_copy = obs_venv.copy()
                    for i in range(self.n_envs):
                        if truncated_venv[i]:
                            obs_venv_copy["state"][i] = info_venv[i]["final_obs"][
                                "state"
                            ]
                    obs_buffer.append(prev_obs_venv["state"])
                    next_obs_buffer.append(obs_venv_copy["state"])
                    action_buffer.append(action_venv)
                    reward_buffer.append(reward_venv * self.scale_reward_factor)
                    terminated_buffer.append(terminated_venv)


                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0



                print("Different 9 changed: remove cnt_episode")

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))

            if len(episodes_start_end) > 0:
                print("len(episodes_start_end) > 0")
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                print("len(episodes_start_end) == 0")
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0

            print("Different 10 changed: update models condition")

            if not eval_mode:
                num_batch = int(
                    self.n_steps * self.n_envs / self.batch_size * self.replay_ratio
                )

                obs_trajs = np.array(deepcopy(obs_buffer))
                action_trajs = np.array(deepcopy(action_buffer))
                next_obs_trajs = np.array(deepcopy(next_obs_buffer))
                reward_trajs = np.array(deepcopy(reward_buffer))
                terminated_trajs = np.array(deepcopy(terminated_buffer))

                # flatten
                obs_trajs = einops.rearrange(
                    obs_trajs,
                    "s e h d -> (s e) h d",
                )
                next_obs_trajs = einops.rearrange(
                    next_obs_trajs,
                    "s e h d -> (s e) h d",
                )
                action_trajs = einops.rearrange(
                    action_trajs,
                    "s e h d -> (s e) h d",
                )
                reward_trajs = reward_trajs.reshape(-1)
                terminated_trajs = terminated_trajs.reshape(-1)
                
                for _ in range(num_batch):


                    time3 = time.time()

                    # Sample batch
                    inds = np.random.choice(len(obs_trajs), self.batch_size)
                    obs_b = torch_tensor_float( torch_from_numpy(obs_trajs[inds]) )
                    next_obs_b = (
                        torch_tensor_float( torch_from_numpy(next_obs_trajs[inds]) )
                    )
                    actions_b = (
                        torch_tensor_float( torch_from_numpy(action_trajs[inds]) )
                    )
                    rewards_b = (
                        torch_tensor_float( torch_from_numpy(reward_trajs[inds]) )
                    )
                    terminated_b = (
                        torch_tensor_float( torch_from_numpy(terminated_trajs[inds]) )
                    )
                        
                    alpha = torch_tensor_item( torch_exp(self.log_alpha) )


                    print("Enter loss critic")
                    with tf.GradientTape(persistent = True) as tape:
                        loss_critic1, loss_critic2 = self.model.loss_critic(
                            {"state": obs_b},
                            {"state": next_obs_b},
                            actions_b,
                            rewards_b,
                            terminated_b,
                            self.gamma,
                            alpha,
                        )

                    tf_Q1_gradients = tape.gradient(loss_critic1, self.model.critic.Q1.trainable_variables)
                    tf_Q2_gradients = tape.gradient(loss_critic2, self.model.critic.Q2.trainable_variables)


                    torch_nn_utils_clip_grad_norm_and_step(
                        self.model.critic.Q1.trainable_variables,
                        self.critic1_optimizer,
                        max_norm=1.0,
                        grads = tf_Q1_gradients,
                        norm_type=2,
                    )

                    torch_nn_utils_clip_grad_norm_and_step(
                        self.model.critic.Q2.trainable_variables,
                        self.critic2_optimizer,
                        max_norm=1.0,
                        grads = tf_Q2_gradients,
                        norm_type=2,
                    )

                    time4 = time.time()
                    elapsed_time = time4 - time3
                    print(f"Elapsed time: single loss critic {elapsed_time:.4f} seconds")

                    
                    if self.model.step % self.model.delay_update == 0:
                        time5 = time.time()

                        print("Enter loss actor")
                        with tf.GradientTape() as tape:
                            
                            loss_actor = self.model.loss_actor(
                                {"state": obs_b},
                                alpha,
                            )

                        tf_gradients_actor = tape.gradient(loss_actor, self.model.actor.trainable_variables)


                        torch_nn_utils_clip_grad_norm_and_step(
                            self.model.actor.trainable_variables,
                            self.actor_optimizer,
                            max_norm=1.0,
                            grads = tf_gradients_actor,
                            norm_type=2,
                        )


                        time6 = time.time()
                        elapsed_time = time6 - time5
                        print(f"Elapsed time: single loss_actor {elapsed_time:.4f} seconds")


                    if self.model.step % self.model.delay_alpha_update == 0:
                        time7 = time.time()


                        print("Enter loss temperature")

                        with tf.GradientTape() as tape:
                            # Update temperature parameter
                            loss_alpha = self.model.loss_temperature(
                                {"state": obs_b},
                                torch_exp( self.log_alpha ),  # with grad
                                self.target_entropy,
                            )
                        

                        tf_alpha_gradients = tape.gradient(loss_alpha, [self.log_alpha])


                        torch_nn_utils_clip_grad_norm_and_step(
                            [self.log_alpha],
                            self.log_alpha_optimizer,
                            max_norm=1.0,
                            grads = tf_alpha_gradients,
                            norm_type=2,
                        )

                        time8 = time.time()
                        elapsed_time = time8 - time7
                        print(f"Elapsed time: single loss_temperature {elapsed_time:.4f} seconds")


                    if self.model.step % self.model.delay_update == 0:
                        time9 = time.time()

                        self.model.update_target_critic(self.model.tau)

                        time10 = time.time()
                        elapsed_time = time10 - time9
                        print(f"Elapsed time: update_target_critic {elapsed_time:.4f} seconds")

                    self.model.step=self.model.step + 1




            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )

            print("Different 12 changed: output")
            if self.itr % self.log_freq == 0:
                end_time = timer()
                run_results[-1]["time"] = end_time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f} | num episode - eval: {num_episode_finished:8.4f}"
                    )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss actor {torch_tensor_item(loss_actor):8.4f} | loss critic1 {torch_tensor_item(loss_critic1):8.4f} | loss critic2 {torch_tensor_item(loss_critic2):8.4f} | reward {avg_episode_reward:8.4f} | num episode - train: {num_episode_finished:8.4f}"
                    )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward            # # if self.itr % self.log_freq == 0 and self.itr > self.n_explore_steps:
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1

