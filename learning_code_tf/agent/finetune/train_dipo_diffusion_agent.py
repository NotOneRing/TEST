"""
Model-free online RL with DIffusion POlicy (DIPO)

Applies action gradient to perturb actions towards maximizer of Q-function.

a_t <- a_t + \eta * \grad_a Q(s, a)

Do not support pixel input right now.

"""

import os
import pickle
import numpy as np

import logging
import wandb

log = logging.getLogger(__name__)
from util.timer import Timer
from collections import deque
from agent.finetune.train_agent import TrainAgent



import tensorflow as tf

from util.torch_to_tf import torch_from_numpy, torch_min, \
    torch_tensor_float, torch_reshape, torch_nn_utils_clip_grad_norm_and_step, torch_clamp,\
    torch_tensor_requires_grad_, torch_sum, torch_tensor_detach, \
    torch_no_grad, torch_nn_utils_clip_grad_norm_and_step

from util.torch_to_tf import tf_CosineAnnealingWarmupRestarts, torch_optim_AdamW, torch_optim_Adam


class TrainDIPODiffusionAgent(TrainAgent):

    def __init__(self, cfg):
        print("train_dipo_diffusion_agent.py: TrainDIPODiffusionAgent.__init__()")

        super().__init__(cfg)

        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Wwarm up period for critic before actor updates
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # use cosine scheduler with linear warmup
        self.actor_lr_scheduler = tf_CosineAnnealingWarmupRestarts(
            # self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        # Optimizer
        self.actor_optimizer = torch_optim_AdamW(
            # self.model.actor.parameters(),
            self.model.actor.trainable_variables,
            lr=self.actor_lr_scheduler,
            weight_decay=cfg.train.actor_weight_decay,
        )
        
        self.critic_lr_scheduler = tf_CosineAnnealingWarmupRestarts(
            # self.critic_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        self.critic_optimizer = torch_optim_AdamW(
            # self.model.critic.parameters(),
            self.model.critic.trainable_variables,
            # lr=cfg.train.critic_lr,
            lr = self.critic_lr_scheduler,
            weight_decay=cfg.train.critic_weight_decay,
        )

        # target update rate
        self.target_ema_rate = cfg.train.target_ema_rate

        # Buffer size
        self.buffer_size = cfg.train.buffer_size

        # Action gradient scaling
        self.action_lr = cfg.train.action_lr

        # Updates
        self.replay_ratio = cfg.train.replay_ratio

        # Scaling reward
        self.scale_reward_factor = cfg.train.scale_reward_factor

        # Apply action gradient many steps
        self.action_gradient_steps = cfg.train.action_gradient_steps

        # Max grad norm for action
        self.action_grad_norm = self.action_dim * self.act_steps * 0.1

    def run(self):
        print("train_dipo_diffusion_agent.py: TrainDIPODiffusionAgent.run()")

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
            
            # self.model.eval() if eval_mode else self.model.train()
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
                # with torch.no_grad():
                with torch_no_grad() as tape:
                    cond = {
                        "state": torch_tensor_float( torch_from_numpy(prev_obs_venv["state"]) )
                        # .float()
                        # .to(self.device)
                    }
                    samples = (
                        self.model(
                            cond=cond,
                            deterministic=eval_mode,
                        ).numpy()
                        # .cpu()
                        # .numpy()
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
                    for i in range(self.n_envs):
                        obs_buffer.append(prev_obs_venv["state"][i])
                        if truncated_venv[i]:  # truncated
                            next_obs_buffer.append(info_venv[i]["final_obs"]["state"])
                        else:
                            next_obs_buffer.append(obs_venv["state"][i])
                        action_buffer.append(action_venv[i])
                    reward_buffer.extend(
                        (reward_venv * self.scale_reward_factor).tolist()
                    )
                    terminated_buffer.extend(terminated_venv.tolist())

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
                # only worth converting first with parallel envs - large number of updates below
                obs_array = np.array(obs_buffer)
                next_obs_array = np.array(next_obs_buffer)
                action_array = np.array(action_buffer)
                reward_array = np.array(reward_buffer)
                terminated_array = np.array(terminated_buffer)

                # Critic learning
                for _ in range(num_batch):
                    inds = np.random.choice(len(obs_buffer), self.batch_size)
                    obs_b = torch_tensor_float( torch_from_numpy(obs_array[inds]) )
                                            #    .float().to(self.device)
                    next_obs_b = (
                        torch_tensor_float( torch_from_numpy(next_obs_array[inds]) )
                        # .float().to(self.device)
                    )
                    actions_b = (
                        torch_tensor_float( torch_from_numpy(action_array[inds]) )
                        # .float().to(self.device)
                    )
                    rewards_b = (
                        torch_tensor_float( torch_from_numpy(reward_array[inds]) )
                        # .float().to(self.device)
                    )
                    terminated_b = (
                        torch_tensor_float(  torch_from_numpy(terminated_array[inds]) )
                        # .float().to(self.device)
                    )

                    with tf.GradientTape() as tape:

                        # Update critic
                        loss_critic = self.model.loss_critic(
                            {"state": obs_b},
                            {"state": next_obs_b},
                            actions_b,
                            rewards_b,
                            terminated_b,
                            self.gamma,
                        )

                    # tf_gradients = tape.gradient(loss_critic, self.model.critic.trainable_variables)                    
                    # self.critic_optimizer.step(tf_gradients)


                    tf_critic_gradients = tape.gradient(loss_critic, self.model.critic.trainable_variables)                        
                    zip_gradients_critic_params = zip(tf_critic_gradients, self.model.critic.trainable_variables)
                    self.critic_optimizer.apply_gradients(zip_gradients_critic_params)



                    # Actor learning
                    loss_actor = 0.0
                    if self.itr >= self.n_critic_warmup_itr:
                        inds = np.random.choice(len(obs_buffer), self.batch_size)
                        obs_b = (
                            torch_tensor_float( torch_from_numpy(obs_array[inds]) )
                            # .to(self.device)
                        )
                        # actions_b = (
                        #     torch_tensor_float( torch_from_numpy(action_array[inds]) )
                        #     # .to(self.device)
                        # )

                        # # get Q-perturbed actions by optimizing
                        # actions_flat = torch_reshape( actions_b, len(actions_b), -1)
                        
                        actions_b = tf.Variable(
                            torch_tensor_float( torch_from_numpy(action_array[inds]) )
                            # .to(self.device)
                        )

                        # get Q-perturbed actions by optimizing
                        actions_flat = torch_reshape( actions_b, len(actions_b), -1)


                        actions_optim = torch_optim_Adam(
                            [actions_flat], lr=self.action_lr, eps=1e-5
                        )

                        for _ in range(self.action_gradient_steps):
                            
                            # actions_flat.requires_grad_(True)
                            actions_flat = torch_tensor_requires_grad_(actions_flat, True)

                            with tf.GradientTape() as tape:
                                
                                print("self.model = ", self.model)
                                print("self.model.critic = ", self.model.critic)
                                
                                q_values_1, q_values_2 = self.model.critic(
                                    {"state": obs_b}, actions_flat
                                )
                                q_values = torch_min(q_values_1, other=q_values_2)
                                action_opt_loss = - torch_sum(q_values)
                            # .sum()

                            tf_gradients = tape.gradient(action_opt_loss, [actions_flat])

                            # actions_optim.zero_grad()
                            # action_opt_loss.backward(torch_ones_like(action_opt_loss))



                            print("q_values = ", q_values)
                            print("q_values.shape = ", q_values.shape)
                            print("action_opt_loss = ", action_opt_loss)
                            print("action_opt_loss.shape = ", action_opt_loss.shape)
                            print("actions_flat = ", actions_flat)
                            print("actions_flat.shape = ", actions_flat.shape)
                            print("tf_gradients = ", tf_gradients)



                            torch_nn_utils_clip_grad_norm_and_step(
                                [actions_flat],
                                actions_optim,
                                max_norm=self.action_grad_norm,
                                grads = tf_gradients,
                                norm_type=2,
                            )
                            # two into one
                            # actions_optim.step(tf_gradients)

                            # actions_flat.requires_grad_(False)
                            torch_tensor_requires_grad_(actions_flat, False)
                            
                            actions_flat = torch_clamp(actions_flat, -1.0, 1.0)

                        guided_action = torch_reshape( actions_flat,
                            len(actions_flat), self.horizon_steps, self.action_dim
                        )
                        # guided_action_np = torch_tensor_detach( guided_action.numpy()
                        #                                     #    .detach()
                        #                                     #    .cpu().numpy()
                        #                                        )
                        guided_action_np = guided_action.numpy()
                        

                        # Add back to buffer
                        action_array[inds] = guided_action_np

                        with tf.GradientTape() as tape:
                            # Update policy with collected trajectories
                            loss_actor = self.model.loss_ori(
                                training,
                                # guided_action.detach()
                                torch_tensor_detach(guided_action),
                                {"state": obs_b}
                            )
                        
                        # tf_gradients = tape.gradient(loss_actor, self.model.actor.trainable_variables)

                        tf_actor_gradients = tape.gradient(loss_actor, self.model.actor.trainable_variables)                        
                        zip_gradients_actor_params = zip(tf_actor_gradients, self.model.actor.trainable_variables)

                        # self.actor_optimizer.zero_grad()
                        # loss_actor.backward()

                        if self.max_grad_norm is not None:
                            torch_nn_utils_clip_grad_norm_and_step(
                                # self.model.actor.parameters()
                                self.model.actor.trainable_variables,
                                self.actor_optimizer,
                                self.max_grad_norm,
                                # tf_gradients
                                tf_actor_gradients
                            )
                        else:
                            # self.actor_optimizer.step(tf_gradients)
                            self.actor_optimizer.apply_gradients(zip_gradients_actor_params)


                    # Update target critic and actor
                    self.model.update_target_critic(self.target_ema_rate)
                    self.model.update_target_actor(self.target_ema_rate)

                # convert back to buffer
                action_buffer = deque(
                    [action for action in action_array], maxlen=self.buffer_size
                )

            # Update lr
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model_dipo()

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
                    # log.info(
                    #     f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    # )
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f} | num episode - eval: {num_episode_finished:8.4f}"
                    )
                    # if self.use_wandb:
                    #     wandb.log(
                    #         {
                    #             "success rate - eval": success_rate,
                    #             "avg episode reward - eval": avg_episode_reward,
                    #             "avg best reward - eval": avg_best_reward,
                    #             "num episode - eval": num_episode_finished,
                    #         },
                    #         step=self.itr,
                    #         commit=False,
                    #     )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    # log.info(
                    #     f"{self.itr}: step {cnt_train_step:8d} | loss actor {loss_actor:8.4f} | loss - critic {loss_critic:8.4f} | reward {avg_episode_reward:8.4f} | t:{time:8.4f}"
                    # )
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss actor {loss_actor:8.4f} | loss - critic {loss_critic:8.4f} | reward {avg_episode_reward:8.4f} | t:{time:8.4f} | num episode - train: {num_episode_finished:8.4f}"
                    )

                    # if self.use_wandb:
                    #     wandb_log = {
                    #         "total env step": cnt_train_step,
                    #         "loss - critic": loss_critic,
                    #         "avg episode reward - train": avg_episode_reward,
                    #         "num episode - train": num_episode_finished,
                    #     }
                    #     # if type(loss_actor) == torch.Tensor:
                    #     if isinstance(loss_actor, tf.tensor):
                    #         wandb_log["loss - actor"] = loss_actor
                    #     wandb.log(wandb_log, step=self.itr, commit=True)
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
