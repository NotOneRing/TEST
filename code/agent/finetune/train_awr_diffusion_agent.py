"""
Advantage-weighted regression (AWR) for diffusion policy.

Advantage = discounted-reward-to-go - V(s)

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



from util.torch_to_tf import torch_from_numpy, torch_tensor_float, torch_flatten, torch_tensor_float, torch_exp, torch_clamp, torch_mean\
, torch_optim_AdamW, torch_nn_utils_clip_grad_norm_and_step, torch_std, torch_tensor_float, CosineAWR,\
torch_tensor_detach, torch_no_grad




def td_values(
    states,
    rewards,
    terminateds,
    state_values,
    gamma=0.99,
    alpha=0.95,
    lam=0.95,
):
    """
    Gives a list of TD estimates for a given list of samples from an RL environment.
    The TD(λ) estimator is used for this computation.

    :param replay_buffers: The replay buffers filled by exploring the RL environment.
    Includes: states, rewards, "final state?"s.
    :param state_values: The currently estimated state values.
    :return: The TD estimates.
    """

    print("train_awr_diffusion_agent.py: td_values()")


    sample_count = len(states)
    tds = np.zeros_like(state_values, dtype=np.float32)
    next_value = state_values[-1].copy()
    next_value[terminateds[-1]] = 0.0

    val = 0.0
    for i in range(sample_count - 1, -1, -1):

        # get next_value for vectorized
        if i < sample_count - 1:
            next_value = state_values[i + 1]
            next_value = next_value * (1 - terminateds[i])

        state_value = state_values[i]
        error = rewards[i] + gamma * next_value - state_value
        val = alpha * error + gamma * lam * (1 - terminateds[i]) * val

        tds[i] = val + state_value
    return tds


class TrainAWRDiffusionAgent(TrainAgent):

    def __init__(self, cfg):

        print("train_awr_diffusion_agent.py: TrainAWRDiffusionAgent.__init__()")

        super().__init__(cfg)
        self.logprob_batch_size = cfg.train.get("logprob_batch_size", 10000)

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
            self.model.critic.trainable_variables,
            lr = self.critic_lr_scheduler,
            weight_decay=cfg.train.critic_weight_decay,
        )

        # Buffer size
        self.buffer_size = cfg.train.buffer_size

        # Reward exponential
        self.beta = cfg.train.beta

        # Max weight for AWR
        self.max_adv_weight = cfg.train.max_adv_weight

        # Scaling reward
        self.scale_reward_factor = cfg.train.scale_reward_factor

        # Updates
        self.replay_ratio = cfg.train.replay_ratio
        self.critic_update_ratio = cfg.train.critic_update_ratio

    def run(self):

        print("train_awr_diffusion_agent.py: TrainAWRDiffusionAgent.run()")

        # make a FIFO replay buffer for obs, action, and reward
        obs_buffer = deque(maxlen=self.buffer_size)
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
                        "state": torch_tensor_float( torch_from_numpy( prev_obs_venv["state"] ) )
                    }
                    samples = (
                        self.model(
                            cond=cond,
                            deterministic=eval_mode,
                        ).numpy()
                    )
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
                    obs_buffer.append(prev_obs_venv["state"])
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
                obs_trajs = np.array(deepcopy(obs_buffer))  # assume only state
                reward_trajs = np.array(deepcopy(reward_buffer))
                terminated_trajs = np.array(deepcopy(terminated_buffer))
                obs_t = einops.rearrange(
                    torch_tensor_float( torch_from_numpy(obs_trajs) ) 
                    ,
                    "s e h d -> (s e) h d",
                )

                values_trajs = np.array(
                    self.model.critic({"state": obs_t}).numpy()
                ).reshape(-1, self.n_envs)

                td_trajs = td_values(
                    obs_trajs, reward_trajs, terminated_trajs, values_trajs
                )

                td_t = torch_tensor_float( torch_flatten( torch_from_numpy(td_trajs) ) )

                # Update critic
                num_batch = int(
                    self.n_steps * self.n_envs / self.batch_size * self.replay_ratio
                )
                for _ in range(num_batch // self.critic_update_ratio):
                    inds = np.random.choice(len(obs_trajs), self.batch_size)


                    obs_result = tf.gather(obs_t, inds, axis=0)
                    td_result = tf.gather(td_t, inds, axis=0)

                    with tf.GradientTape() as tape:
                        loss_critic = self.model.loss_critic(
                            {"state": obs_result}, td_result
                        )


                    tf_gradients_critic = tape.gradient(loss_critic, self.model.critic.trainable_variables)                        
                    zip_gradients_critic_params = zip(tf_gradients_critic, self.model.critic.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip_gradients_critic_params)


                # Update policy - use a new copy of data
                obs_trajs = np.array(deepcopy(obs_buffer))
                samples_trajs = np.array(deepcopy(action_buffer))
                reward_trajs = np.array(deepcopy(reward_buffer))
                terminated_trajs = np.array(deepcopy(terminated_buffer))
                obs_t = einops.rearrange(
                    torch_tensor_float( torch_from_numpy(obs_trajs) ),
                    "s e h d -> (s e) h d",
                )
                values_trajs = np.array(
                    self.model.critic({"state": obs_t}).numpy()
                ).reshape(-1, self.n_envs)

                td_trajs = td_values(
                    obs_trajs, reward_trajs, terminated_trajs, values_trajs
                )
                advantages_trajs = td_trajs - values_trajs

                # flatten
                obs_trajs = einops.rearrange(
                    obs_trajs,
                    "s e h d -> (s e) h d",
                )
                samples_trajs = einops.rearrange(
                    samples_trajs,
                    "s e h d -> (s e) h d",
                )
                advantages_trajs = einops.rearrange(
                    advantages_trajs,
                    "s e -> (s e)",
                )

                for _ in range(num_batch):

                    # Sample batch
                    inds = np.random.choice(len(obs_trajs), self.batch_size)
                    obs_b = {
                        "state": torch_tensor_float( torch_from_numpy(obs_trajs[inds]) )
                    }
                    actions_b = (
                        torch_tensor_float( torch_from_numpy(samples_trajs[inds]) )
                    )
                    advantages_b = (
                        torch_tensor_float( torch_from_numpy(advantages_trajs[inds]) )
                    )
                    advantages_b = (advantages_b - torch_mean( advantages_b) ) / (
                        torch_std( advantages_b )
                        + 1e-6
                    )
                    
                    advantages_b_scaled = torch_exp(self.beta * advantages_b)
                    
                    advantages_b_scaled = torch_clamp( advantages_b_scaled, max=self.max_adv_weight)

                    with tf.GradientTape() as tape:
                        # Update policy with collected trajectories
                        loss_actor = self.model.loss_ori(
                            training,
                            actions_b,
                            obs_b,
                            torch_tensor_detach( advantages_b_scaled )
                            ,
                        )

                    tf_gradients_actor = tape.gradient(loss_actor, self.model.actor.trainable_variables)                        
                    zip_gradients_actor_params = zip(tf_gradients_actor, self.model.actor.trainable_variables)


                    if self.itr >= self.n_critic_warmup_itr:
                        if self.max_grad_norm is not None:
                            torch_nn_utils_clip_grad_norm_and_step(
                                self.model.actor.trainable_variables,
                                self.actor_optimizer,
                                self.max_grad_norm,
                                tf_gradients_actor
                            )
                        else:
                            self.actor_optimizer.apply_gradients(zip_gradients_actor_params)

            # Update lr
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model_awr()

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
                        f"{self.itr}: step {cnt_train_step:8d} | loss actor {loss_actor:8.4f} | loss critic {loss_critic:8.4f} | reward {avg_episode_reward:8.4f} | num episode - train: {num_episode_finished:8.4f} | t:{time:8.4f}"
                    )
                    
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1






















