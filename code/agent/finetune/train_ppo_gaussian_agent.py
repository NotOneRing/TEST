"""
PPO training for Gaussian/GMM policy.

"""

import os
import pickle
import einops
import numpy as np



# import torch
import tensorflow as tf


import logging
import wandb
import math

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_agent import TrainPPOAgent


from util.torch_to_tf import torch_no_grad, torch_from_numpy,\
torch_tensor_float, torch_tensor_transpose, torch_split, \
torch_reshape, torch_tensor, torch_tensor_float,\
torch_randperm, torch_nn_utils_clip_grad_norm_and_step



class TrainPPOGaussianAgent(TrainPPOAgent):

    def __init__(self, cfg):

        print("train_ppo_gaussian_agent.py: TrainPPOGaussianAgent.__init__()")

        super().__init__(cfg)

    def run(self):
        print("train_ppo_gaussian_agent.py: TrainPPOGaussianAgent.run()")

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
                training = False
            else:
                training = True

            last_itr_eval = eval_mode

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv

            # Holder
            obs_trajs = {
                "state": np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim)
                )
            }
            samples_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            if self.save_full_observations:
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
                )

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
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        training = training
                    )
                    # output_venv = samples.cpu().numpy()
                    output_venv = samples.numpy()

                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                if self.save_full_observations:  # state-only
                    obs_full_venv = np.array(
                        [info["full_obs"]["state"] for info in info_venv]
                    )  # n_envs x act_steps x obs_dim
                    obs_full_trajs = np.vstack(
                        (obs_full_trajs, 
                         obs_full_venv.transpose(1, 0, 2)
                         )
                    )
                obs_trajs["state"][step] = prev_obs_venv["state"]
                samples_trajs[step] = output_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

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
                if (
                    self.furniture_sparse_reward
                ):  # only for furniture tasks, where reward only occurs in one env step
                    episode_best_reward = episode_reward
                else:
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
                # with torch.no_grad():
                with torch_no_grad() as tape:
                    obs_trajs["state"] = (
                        torch_tensor_float( torch_from_numpy(obs_trajs["state"]) )
                    )

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    num_split = math.ceil(
                        self.n_envs * self.n_steps / self.logprob_batch_size
                    )
                    obs_ts = [{} for _ in range(num_split)]
                    obs_k = einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                    obs_ts_k = torch_split(obs_k, self.logprob_batch_size, dim=0)
                    for i, obs_t in enumerate(obs_ts_k):
                        obs_ts[i]["state"] = obs_t
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:
                        # values = self.model.critic(obs).cpu().numpy().flatten()
                        values = self.model.critic(obs).numpy().flatten()

                        values_trajs = np.vstack(
                            (values_trajs, values.reshape(-1, self.n_envs))
                        )
                    
                    samples_t = einops.rearrange(
                        torch_tensor_float( torch_from_numpy(samples_trajs) ),
                        #    .float().to(self.device),
                        "s e h d -> (s e) h d",
                    )

                    samples_ts = torch_split(samples_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty((0))
                    for obs_t, samples_t in zip(obs_ts, samples_ts):
                        logprobs = (
                            # self.model.get_logprobs(obs_t, samples_t)[0].cpu().numpy()
                            self.model.get_logprobs(obs_t, samples_t)[0].numpy()
                        )
                        logprobs_trajs = np.concatenate(
                            (
                                logprobs_trajs,
                                logprobs.reshape(-1),
                            )
                        )

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
                    obs_venv_ts = {
                        "state": torch_tensor_float( torch_from_numpy(obs_venv["state"]) )
                        # .float()
                        # .to(self.device)
                    }
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = (
                                # self.model.critic(obs_venv_ts)
                                # .reshape(1, -1)
                                torch_reshape( self.model.critic(obs_venv_ts), 1, -1).numpy()
                                # .cpu()
                                # .numpy()
                            )
                        else:
                            nextvalues = values_trajs[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        # delta = r + gamma*V(st+1) - V(st)
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nonterminal
                            - values_trajs[t]
                        )
                        # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                        advantages_trajs[t] = lastgaelam = (
                            delta
                            + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs

                # k for environment step
                obs_k = {
                    "state": einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                }
                samples_k = einops.rearrange(
                    torch_tensor_float( torch_tensor(samples_trajs ) ),
                    "s e h d -> (s e) h d",
                )
                returns_k = (
                    torch_reshape( torch_tensor_float( torch_tensor(returns_trajs ) ), -1)
                )
                values_k = (
                    torch_reshape( torch_tensor_float( torch_tensor(values_trajs ) ), -1)
                )
                advantages_k = (
                    torch_reshape( torch_tensor_float( torch_tensor(advantages_trajs ) ), -1)
                )
                logprobs_k = torch_tensor_float( torch_tensor(logprobs_trajs ) )

                # Update policy and critic
                total_steps = self.n_steps * self.n_envs
                clipfracs = []
                for update_epoch in range(self.update_epochs):

                    # for each epoch, go through all data in batches
                    flag_break = False
                    inds_k = torch_randperm(total_steps)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        

                        # obs_b = {"state": obs_k["state"][inds_b]}
                        temp_result = tf.gather(obs_k["state"], inds_b, axis=0)
                        obs_b = {"state": temp_result}


                        # samples_b = samples_k[inds_b]
                        # returns_b = returns_k[inds_b]
                        # values_b = values_k[inds_b]
                        # advantages_b = advantages_k[inds_b]
                        # logprobs_b = logprobs_k[inds_b]

                        samples_b = tf.gather(samples_k, inds_b, axis=0)
                        returns_b = tf.gather(returns_k, inds_b, axis=0)
                        values_b = tf.gather(values_k, inds_b, axis=0)
                        advantages_b = tf.gather(advantages_k, inds_b, axis=0)
                        logprobs_b = tf.gather(logprobs_k, inds_b, axis=0)


                        with tf.GradientTape(persistent=True) as tape:

                            # get loss
                            (
                                pg_loss,
                                entropy_loss,
                                v_loss,
                                clipfrac,
                                approx_kl,
                                ratio,
                                bc_loss,
                                std,
                            ) = self.model.loss_ori(
                                obs_b,
                                samples_b,
                                returns_b,
                                values_b,
                                advantages_b,
                                logprobs_b,
                                use_bc_loss=self.use_bc_loss,
                            )
                            loss = (
                                pg_loss
                                + entropy_loss * self.ent_coef
                                + v_loss * self.vf_coef
                                + bc_loss * self.bc_loss_coeff
                            )
                            clipfracs += [clipfrac]

                        tf_gradients_actor_ft = tape.gradient(loss, self.model.actor_ft.trainable_variables)

                        tf_gradients_critic = tape.gradient(loss, self.model.critic.trainable_variables)

                        zip_tf_gradients_actor_ft = zip(tf_gradients_actor_ft, self.model.actor_ft.trainable_variables)

                        zip_tf_gradients_critic = zip(tf_gradients_critic, self.model.critic.trainable_variables)


                        # # update policy and critic
                        # self.actor_optimizer.zero_grad()
                        # self.critic_optimizer.zero_grad()
                        # loss.backward()
                        if self.itr >= self.n_critic_warmup_itr:
                            if self.max_grad_norm is not None:
                                torch_nn_utils_clip_grad_norm_and_step(
                                    self.model.actor_ft.trainable_variables,
                                    self.actor_optimizer,
                                    self.max_grad_norm,
                                    tf_gradients_actor_ft
                                )
                            else:
                                self.actor_optimizer.apply_gradients(zip_tf_gradients_actor_ft)
                        #         self.actor_optimizer.step(tf_gradients_actor_ft)
                        # self.critic_optimizer.step(tf_gradients_critic)

                        self.critic_optimizer.apply_gradients(zip_tf_gradients_critic)


                        log.info(
                            f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                        )

                        # Stop gradient update if KL difference reaches target
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break

                # Explained variation of future rewards using value function
                # y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()

                y_pred, y_true = values_k.numpy(), returns_k.numpy()

                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

            # Plot state trajectories (only in D3IL)
            if (
                self.itr % self.render_freq == 0
                and self.n_render > 0
                and self.traj_plotter is not None
            ):
                self.traj_plotter(
                    obs_full_trajs=obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )

            # Update lr
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model_gaussian_or_gmm()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["action_trajs"] = samples_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
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
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} "
                        f"| pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} "
                        f"| ent {-entropy_loss:8.4f} | reward {avg_episode_reward:8.4f} "
                        f"| t:{time:8.4f} "
                        f"| std:{std:8.4f} | approx kl:{approx_kl:8.4f} | ratio:{ratio:8.4f} "
                        f"| clipfrac:{np.mean(clipfracs):8.4f} | explained variance:{explained_var:8.4f} "
                        f"| num episode - train: {num_episode_finished:8.4f}"
                    )
                    # if self.use_wandb:
                    #     wandb.log(
                    #         {
                    #             "total env step": cnt_train_step,
                    #             "loss": loss,
                    #             "pg loss": pg_loss,
                    #             "value loss": v_loss,
                    #             "entropy": -entropy_loss,
                    #             "std": std,
                    #             "approx kl": approx_kl,
                    #             "ratio": ratio,
                    #             "clipfrac": np.mean(clipfracs),
                    #             "explained variance": explained_var,
                    #             "avg episode reward - train": avg_episode_reward,
                    #             "num episode - train": num_episode_finished,
                    #         },
                    #         step=self.itr,
                    #         commit=True,
                    #     )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1






