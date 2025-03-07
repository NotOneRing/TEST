"""
Original Diffusion Actor-Critic with Entropy Regulator(DACER) agent training script.

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

# from util.torch_to_tf import torch_tensor, torch_from_numpy, torch_tensor_float, torch_exp
# from util.torch_to_tf import torch_no_grad, torch_optim_Adam, torch_exp, torch_tensor_item, torch_tensor_requires_grad_

from util.torch_to_tf import *




class TrainOriginalDacerAgent(TrainAgent):
    def __init__(self, cfg):
        print("train_sac_agent.py: TrainSACAgent.__init__()")

        super().__init__(cfg)
                
        
        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma



        # # Optimizer
        # self.actor_optimizer = torch_optim_Adam(
        #     self.model.actor.trainable_variables,
        #     lr=cfg.train.actor_lr,
        # )

        # self.critic1_optimizer = torch_optim_Adam(
        #     self.model.critic.Q1.trainable_variables,
        #     lr=cfg.train.critic_lr,
        # )

        # self.critic2_optimizer = torch_optim_Adam(
        #     self.model.critic.Q2.trainable_variables,
        #     lr=cfg.train.critic_lr,
        # )


        # Optimizer
        self.actor_lr_scheduler = CosineAWR(
            # self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_q1_lr_scheduler = CosineAWR(
            # self.critic_v_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_q2_lr_scheduler = CosineAWR(
            # self.critic_q_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )


        self.actor_optimizer = torch_optim_AdamW(
            self.model.actor.trainable_variables,
            lr = self.actor_lr_scheduler,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.critic1_optimizer = torch_optim_AdamW(
            self.model.critic.Q1.trainable_variables,
            lr = self.critic_q1_lr_scheduler,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic2_optimizer = torch_optim_AdamW(
            self.model.critic.Q2.trainable_variables,
            lr = self.critic_q2_lr_scheduler,
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
            lr=cfg.train.critic_lr,
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
            # eval_mode = self.itr % self.val_freq == 0 and self.itr > self.n_explore_steps and not self.force_train
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train

            print("Different 2 removed: n_steps = ( self.n_steps if not eval_mode else int(1e5) )")
            # n_steps = ( self.n_steps if not eval_mode else int(1e5) )
            

            if eval_mode:
                training=False
            else:
                training=True

            print("Different 3 added: last_itr_eval = eval_mode")
            last_itr_eval = eval_mode



            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) at the beginning
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))

            print("Different 4 changed: if self.reset_at_iteration or eval_mode or self.itr == 0:")
            # if self.reset_at_iteration or eval_mode or self.itr == 0:
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv

            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            # Collect a set of trajectories from env
            print("Different 5 removed: cnt_episode = 0")
            # cnt_episode = 0

            print("Different 6 changed: for step in range(n_steps)")
            # for step in range(n_steps):
            for step in range(self.n_steps):
                
                time1 = time.time()

                print("Different 7 changed: if self.itr < self.n_explore_steps")
                # # Select action
                # if self.itr < self.n_explore_steps:
                #     print("branch1: self.itr < self.n_explore_steps")
                #     action_venv = self.venv.action_space.sample()
                # else:
                #     print("branch2: self.itr >= self.n_explore_steps")
                #     # with torch.no_grad():
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
                # # add to buffer in train mode
                # if not eval_mode:
                #     for i in range(self.n_envs):
                #         obs_buffer.append(prev_obs_venv["state"][i])
                #         if "final_obs" in info_venv[i]:  # truncated
                #             next_obs_buffer.append(info_venv[i]["final_obs"]["state"])
                #         else:  # first obs in new episode
                #             next_obs_buffer.append(obs_venv["state"][i])
                #         action_buffer.append(action_venv[i])
                #     reward_buffer.extend(
                #         (reward_venv * self.scale_reward_factor).tolist()
                #     )
                #     terminated_buffer.extend(terminated_venv.tolist())
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
                # # check if enough eval episodes are done
                # cnt_episode += np.sum(done_venv)
                # if eval_mode and cnt_episode >= self.n_eval_episode:
                #     break

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
            # # Update models
            # if (
            #     not eval_mode
            #     and self.itr > self.n_explore_steps
            #     and self.itr % self.critic_update_freq == 0
            # ):
            #     inds = np.random.choice(len(obs_buffer), self.batch_size, replace=False)
            #     obs_b = (
            #         torch_tensor_float( torch_from_numpy(np.array([obs_buffer[i] for i in inds])) )
            #         # .float()
            #         # .to(self.device)
            #     )
            #     next_obs_b = (
            #         torch_tensor_float( torch_from_numpy(np.array([next_obs_buffer[i] for i in inds])) )
            #         # .float()
            #         # .to(self.device)
            #     )
            #     actions_b = (
            #         torch_tensor_float( torch_from_numpy(np.array([action_buffer[i] for i in inds])) )
            #         # .float()
            #         # .to(self.device)
            #     )
            #     rewards_b = (
            #         torch_tensor_float( torch_from_numpy(np.array([reward_buffer[i] for i in inds])) )
            #         # .float()
            #         # .to(self.device)
            #     )
            #     terminated_b = (
            #         torch_tensor_float( torch_from_numpy(np.array([terminated_buffer[i] for i in inds])) )
            #         # .float()
            #         # .to(self.device)
            #     )

            #     # Update critic
            #     # alpha = self.log_alpha.exp().item()
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

                    zip_gradients_Q1_params = zip(tf_Q1_gradients, self.model.critic.Q1.trainable_variables)
                    zip_gradients_Q2_params = zip(tf_Q2_gradients, self.model.critic.Q2.trainable_variables)

                    # everytime
                    self.critic1_optimizer.apply_gradients(zip_gradients_Q1_params)
                    self.critic2_optimizer.apply_gradients(zip_gradients_Q2_params)

                    time4 = time.time()
                    elapsed_time = time4 - time3
                    print(f"Elapsed time: single loss critic {elapsed_time:.4f} seconds")

                    
                    if self.model.step % self.model.delay_update == 0:
                        time5 = time.time()

                        with tf.GradientTape() as tape:
                            
                            loss_actor = self.model.loss_actor(
                                {"state": obs_b},
                                alpha,
                            )

                        tf_gradients_actor = tape.gradient(loss_actor, self.model.actor.trainable_variables)

                        # print("tf_gradients_actor = ", tf_gradients_actor)

                        zip_tf_gradients_actor_params = zip(tf_gradients_actor, self.model.actor.trainable_variables)


                        # print grad check
                        # for grad, var in zip_tf_gradients_actor_params:
                        #     if grad is None:
                        #         print(f"Gradient for {var} is None")
                        #     else:
                        #         print(f"Gradient for {var}: {grad}")


                        # zipped_gradients_list = list(zip_tf_gradients_actor_params)

                        # assert len(zipped_gradients_list) > 0, "No gradients provided"

                        self.actor_optimizer.apply_gradients(zip_tf_gradients_actor_params)

                        time6 = time.time()
                        elapsed_time = time6 - time5
                        print(f"Elapsed time: single loss_actor {elapsed_time:.4f} seconds")


                    if self.model.step % self.model.delay_alpha_update == 0:
                        time7 = time.time()

                        with tf.GradientTape() as tape:
                            # Update temperature parameter
                            loss_alpha = self.model.loss_temperature(
                                {"state": obs_b},
                                torch_exp( self.log_alpha ),  # with grad
                                self.target_entropy,
                            )
                        

                        tf_alpha_gradients = tape.gradient(loss_alpha, [self.log_alpha])
                        zip_tf_gradients_alpha = zip(tf_alpha_gradients, [self.log_alpha])
                        self.log_alpha_optimizer.apply_gradients(zip_tf_gradients_alpha)

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




                # # Delay update actor
                # loss_actor = 0
                # # if self.itr % self.actor_update_freq == 0:
                # #     for _ in range(2):





            # # Save model
            # if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            #     self.save_model_dacer()


            print("Different 11 changed: update lr scheduler")
            # Update lr
            self.actor_lr_scheduler.step()
            self.critic_q1_lr_scheduler.step()
            self.critic_q2_lr_scheduler.step()


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
                        f"{self.itr}: step {cnt_train_step:8d} | loss actor {loss_actor:8.4f} | loss critic1 {loss_critic1:8.4f} | loss critic2 {loss_critic2:8.4f} | reward {avg_episode_reward:8.4f} | num episode - train: {num_episode_finished:8.4f}"
                    )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward            # # if self.itr % self.log_freq == 0 and self.itr > self.n_explore_steps:
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1


            # if self.itr % self.log_freq == 0:
            #     time = timer()
            #     if eval_mode:
            #         log.info(
            #             f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
            #         )
            #         # if self.use_wandb:
            #         #     wandb.log(
            #         #         {
            #         #             "success rate - eval": success_rate,
            #         #             "avg episode reward - eval": avg_episode_reward,
            #         #             "avg best reward - eval": avg_best_reward,
            #         #             "num episode - eval": num_episode_finished,
            #         #         },
            #         #         step=self.itr,
            #         #         commit=False,
            #         #     )
            #         run_results[-1]["eval_success_rate"] = success_rate
            #         run_results[-1]["eval_episode_reward"] = avg_episode_reward
            #         run_results[-1]["eval_best_reward"] = avg_best_reward
            #     else:
            #         # log.info(
            #         #     f"{self.itr}: step {cnt_train_step:8d} | loss actor {loss_actor:8.4f} | loss critic {loss_critic:8.4f} | reward {avg_episode_reward:8.4f} | alpha {alpha:8.4f} | t {time:8.4f}"
            #         # )
            #         log.info(
            #             f"{self.itr}: step {cnt_train_step:8d} "
            #             f"| loss actor {loss_actor:8.4f} |"
            #             f"| loss critic1 {loss_critic1:8.4f} | loss critic2 {loss_critic2:8.4f} "
            #             f"| entropy coeff: {alpha:8.4f} "
            #             f"| reward {avg_episode_reward:8.4f} "
            #             f"| num episode - train: {num_episode_finished:8.4f} "
            #             f"| t:{time:8.4f}"
            #         )

            #         # if self.use_wandb:
            #         #     wandb_log_dict = {
            #         #         "total env step": cnt_train_step,
            #         #         "loss - critic": loss_critic,
            #         #         "entropy coeff": alpha,
            #         #         "avg episode reward - train": avg_episode_reward,
            #         #         "num episode - train":' num_episode_finished,
            #         #     }
            #         #     if loss_actor is not None:
            #         #         wandb_log_dict["loss - actor"] = loss_actor
            #         #     wandb.log(
            #         #         wandb_log_dict,
            #         #         step=self.itr,
            #         #         commit=True,
            #         #     )
            #         run_results[-1]["train_episode_reward"] = avg_episode_reward
            #     with open(self.result_path, "wb") as f:
            #         pickle.dump(run_results, f)
            # self.itr += 1



