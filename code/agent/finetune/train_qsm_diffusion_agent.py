"""
QSM (Q-Score Matching) for diffusion policy.

Do not support pixel input right now.

"""

import os
import pickle
import einops
import numpy as np


import tensorflow as tf

import logging
import wandb
from copy import deepcopy

log = logging.getLogger(__name__)
from util.timer import Timer
from collections import deque
from agent.finetune.train_agent import TrainAgent

from util.torch_to_tf import CosineAWR

from util.torch_to_tf import torch_no_grad, torch_optim_Adam, torch_optim_AdamW, torch_tensor_float, torch_from_numpy, \
torch_nn_utils_clip_grad_norm_and_step




class TrainQSMDiffusionAgent(TrainAgent):

    def __init__(self, cfg):

        print("train_qsm_diffusion_agent.py: TrainQSMDiffusionAgent.__init__()")

        super().__init__(cfg)

        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Wwarm up period for critic before actor updates
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        self.actor_lr_scheduler = CosineAWR(
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        # Optimizer
        self.actor_optimizer = torch_optim_AdamW(
            self.model.actor.trainable_variables,
            lr = self.actor_lr_scheduler,
            weight_decay=cfg.train.actor_weight_decay,
        )


        self.critic_lr_scheduler = CosineAWR(
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        self.critic_optimizer = torch_optim_AdamW(
            self.model.critic_q.trainable_variables,
            self.critic_lr_scheduler,
            weight_decay=cfg.train.critic_weight_decay,
        )

        # Buffer size
        self.buffer_size = cfg.train.buffer_size

        # Scaling reward
        self.scale_reward_factor = cfg.train.scale_reward_factor

        # Updates
        self.replay_ratio = cfg.train.replay_ratio
        self.critic_tau = cfg.train.critic_tau
        self.q_grad_coeff = cfg.train.q_grad_coeff

    def run(self):
        print("train_qsm_diffusion_agent.py: TrainQSMDiffusionAgent.run()")

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
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:

            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train


            if eval_mode:
                training=False
            else:
                training=True
            
            last_itr_eval = eval_mode

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv
            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch_no_grad() as tape:
                    cond = {
                        "state": torch_tensor_float( torch_from_numpy(prev_obs_venv["state"]) )
                    }
                    samples = (
                        self.model(
                            cond=cond,
                            deterministic=eval_mode,
                        ).numpy()
                    )  # n_env x horizon x act
                action_venv = samples[:, : self.act_steps]

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                reward_trajs[step] = reward_venv
                firsts_trajs[step + 1] = done_venv

                # add to buffer
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
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update models
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
                    inds = np.random.choice(len(obs_trajs), self.batch_size)
                    obs_b = torch_tensor_float(torch_from_numpy(obs_trajs[inds]))
                    next_obs_b = (
                        torch_tensor_float(torch_from_numpy(next_obs_trajs[inds]))
                    )
                    actions_b = (
                        torch_tensor_float(torch_from_numpy(action_trajs[inds]))
                    )
                    rewards_b = (
                        torch_tensor_float(torch_from_numpy(reward_trajs[inds]))
                    )
                    terminated_b = (
                        torch_tensor_float(torch_from_numpy(terminated_trajs[inds]))
                    )


                    with tf.GradientTape() as tape:
                        # update critic q function
                        loss_critic = self.model.loss_critic(
                            {"state": obs_b},
                            {"state": next_obs_b},
                            actions_b,
                            rewards_b,
                            terminated_b,
                            self.gamma,
                        )
                        
                    tf_critic_gradients = tape.gradient(loss_critic, self.model.critic_q.trainable_variables)                        
                    zip_gradients_critic_params = zip(tf_critic_gradients, self.model.critic_q.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip_gradients_critic_params)


                    with tf.GradientTape() as tape:
                        # Update policy with collected trajectories
                        loss_actor = self.model.loss_actor(
                            {"state": obs_b},
                            actions_b,
                            self.q_grad_coeff,
                        )


                    tf_actor_gradients = tape.gradient(loss_actor, self.model.actor.trainable_variables)                        
                    zip_gradients_actor_params = zip(tf_actor_gradients, self.model.actor.trainable_variables)

                    if self.itr >= self.n_critic_warmup_itr:
                        if self.max_grad_norm is not None:
                            torch_nn_utils_clip_grad_norm_and_step(
                                self.model.actor.trainable_variables,
                                self.actor_optimizer,
                                self.max_grad_norm,
                                tf_actor_gradients
                            )
                        else:
                            self.actor_optimizer.apply_gradients(zip_gradients_actor_params)

                    # update target critic
                    self.model.update_target_critic(self.critic_tau)

            # Update lr
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model_qsm()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f} | num episode - eval: {num_episode_finished:8.4f}"
                    )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} \
                        | loss actor {loss_actor:8.4f} | loss critic {loss_critic:8.4f} \
                        | reward {avg_episode_reward:8.4f} \
                        | num episode - train: {num_episode_finished:8.4f} \
                        | t:{time:8.4f}"
                    )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1












