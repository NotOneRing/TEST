"""
PPO training for Gaussian/GMM policy with pixel observations.

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
from agent.finetune.train_ppo_gaussian_agent import TrainPPOGaussianAgent
from model.common.modules import RandomShiftsAug

from util.torch_to_tf import torch_no_grad, torch_from_numpy, torch_tensor_float, torch_split, torch_reshape, torch_tensor, torch_randperm, \
torch_nn_utils_clip_grad_norm_and_step, torch_flatten



class TrainPPOImgGaussianAgent(TrainPPOGaussianAgent):

    def __init__(self, cfg):
        print("train_ppo_gaussian_img_agent.py: TrainPPOImgGaussianAgent.__init__()")

        super().__init__(cfg)

        # Image randomization
        self.augment = cfg.train.augment
        if self.augment:
            self.aug = RandomShiftsAug(pad=4)

        # Set obs dim -  we will save the different obs in batch in a dict
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs.keys()}

        # Gradient accumulation to deal with large GPU RAM usage
        self.grad_accumulate = cfg.train.grad_accumulate

    def run(self):
        print("train_ppo_gaussian_img_agent.py: TrainPPOImgGaussianAgent.run()")

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
            # training = False if eval_mode else traning = True

            if eval_mode:
                training = False
            else:
                traning = True


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
                k: np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dims[k])
                )
                for k in self.obs_dims
            }
            samples_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                # with torch.no_grad():
                with torch_no_grad() as tape:
                    cond = {
                        key: torch_tensor_float( torch_from_numpy(prev_obs_venv[key]) )
                        # .float()
                        # .to(self.device)
                        for key in self.obs_dims
                    }

                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                    )
                    # output_venv = samples.cpu().numpy()
                    output_venv = samples.numpy()
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                for k in obs_trajs:
                    obs_trajs[k][step] = prev_obs_venv[k]
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
                    # apply image randomization
                    obs_trajs["rgb"] = (
                        torch_tensor_float( torch_from_numpy(obs_trajs["rgb"]) )
                        # .float().to(self.device)
                    )
                    obs_trajs["state"] = (
                        torch_tensor_float( torch_from_numpy(obs_trajs["state"]) )
                        # .float().to(self.device)
                    )
                    if self.augment:
                        rgb = einops.rearrange(
                            obs_trajs["rgb"],
                            "s e t c h w -> (s e t) c h w",
                        )
                        rgb = self.aug(rgb)
                        obs_trajs["rgb"] = einops.rearrange(
                            rgb,
                            "(s e t) c h w -> s e t c h w",
                            s=self.n_steps,
                            e=self.n_envs,
                        )

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    num_split = math.ceil(
                        self.n_envs * self.n_steps / self.logprob_batch_size
                    )
                    obs_ts = [{} for _ in range(num_split)]
                    for k in obs_trajs:
                        obs_k = einops.rearrange(
                            obs_trajs[k],
                            "s e ... -> (s e) ...",
                        )
                        obs_ts_k = torch_split(obs_k, self.logprob_batch_size, dim=0)
                        for i, obs_t in enumerate(obs_ts_k):
                            obs_ts[i][k] = obs_t
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:

                        temp_result = self.model.critic(obs, no_augment=True).numpy()
                        temp_result = torch_flatten(temp_result)

                        values = (
                            # self.model.critic(obs, no_augment=True)
                            # .cpu()
                            # .numpy()
                            # .flatten()
                        )
                        values_trajs = np.vstack(
                            (values_trajs, values.reshape(-1, self.n_envs))
                        )
                    samples_t = einops.rearrange(
                        torch_tensor_float(torch_from_numpy(samples_trajs)),
                        "s e h d -> (s e) h d",
                    )
                    samples_ts = torch_split(samples_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty((0))
                    for obs_t, samples_t in zip(obs_ts, samples_ts):
                        logprobs = (
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
                    # obs_venv_ts = {
                    #     key: torch_from_numpy(obs_venv[key]).float().to(self.device)
                    #     for key in self.obs_dims
                    # }

                    obs_venv_ts = {
                        key: torch_tensor_float( torch_from_numpy(obs_venv[key]) )
                        for key in self.obs_dims
                    }

                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = (
                                torch_reshape( self.model.critic(obs_venv_ts, no_augment=True), 1, -1).numpy()
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
                    k: einops.rearrange(
                        obs_trajs[k],
                        "s e ... -> (s e) ...",
                    )
                    for k in obs_trajs
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
                    inds_k = torch_randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        obs_b = {k: obs_k[k][inds_b] for k in obs_k}


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


                        # update policy and critic
                        if (batch + 1) % self.grad_accumulate == 0:
                            if self.itr >= self.n_critic_warmup_itr:
                                if self.max_grad_norm is not None:
                                    torch_nn_utils_clip_grad_norm_and_step(
                                        # self.model.actor_ft.parameters(),
                                        self.model.actor_ft.trainable_variables,
                                        self.actor_optimizer,
                                        self.max_grad_norm,
                                        tf_gradients_actor_ft
                                    )
                                else:
                                    self.actor_optimizer.step(tf_gradients_actor_ft)

                            self.critic_optimizer.step(tf_gradients_critic)


                            log.info(f"run grad update at batch {batch}")
                            log.info(
                                f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                            )

                            # Stop gradient update if KL difference reaches target
                            if (
                                self.target_kl is not None
                                and approx_kl > self.target_kl
                                and self.itr >= self.n_critic_warmup_itr
                            ):
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

            # Update lr
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

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
                    #     f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} 
                    #     | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | 
                    #     reward {avg_episode_reward:8.4f} | t:{time:8.4f}"
                    # )
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} "
                        f"| pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} "
                        f"| bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} "
                        f"| t:{time:8.4f} "
                        f"| std:{std:8.4f} | approx kl:{approx_kl:8.4f} | ratio:{ratio:8.4f}"
                        f"| clipfrac:{np.mean(clipfracs):8.4f} | explained variance:{explained_var:8.4f} "
                        f"| num episode - train: {num_episode_finished:8.4f} "
                        # f"| actor lr : {self.actor_optimizer.param_groups[0]["lr"]:8.4f}, "
                        # f"| critic lr : {self.critic_optimizer.param_groups[0]["lr"]:8.4f} "
                    )
                    # if self.use_wandb:
                    #     wandb.log(
                    #         {
                    #             "total env step": cnt_train_step,
                    #             "loss": loss,
                    #             "pg loss": pg_loss,
                    #             "value loss": v_loss,
                    #             "bc loss": bc_loss,
                    #             "std": std,
                    #             "approx kl": approx_kl,
                    #             "ratio": ratio,
                    #             "clipfrac": np.mean(clipfracs),
                    #             "explained variance": explained_var,
                    #             "avg episode reward - train": avg_episode_reward,
                    #             "num episode - train": num_episode_finished,
                    #             "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                    #             "critic lr": self.critic_optimizer.param_groups[0][
                    #                 "lr"
                    #             ],
                    #         },
                    #         step=self.itr,
                    #         commit=True,
                    #     )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1



























