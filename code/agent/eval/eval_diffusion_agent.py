"""
Evaluate pre-trained/DPPO-fine-tuned diffusion policy.

"""

import os
import numpy as np

import tensorflow as tf
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent

from util.torch_to_tf import torch_no_grad

from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, OUTPUT_POSITIONS, OUTPUT_FUNCTION_HEADER


class EvalDiffusionAgent(EvalAgent):

    def __init__(self, cfg):
        print("eval_diffusion_agent.py: EvalDiffusionAgent.__init__()")

        super().__init__(cfg)

    def run(self):
        print("eval_diffusion_agent.py: EvalDiffusionAgent.run()")

        # Start training loop
        timer = Timer()

        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )

        # Reset env before iteration starts

        training = False

        if OUTPUT_VARIABLES:
            print("self.n_envs = ", self.n_envs)

            print("self.n_render = ", self.n_render)

            print("self.n_steps = ", self.n_steps)

            print("self.act_steps = ", self.act_steps)


        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))

        if self.save_full_observations:  # state-only
            obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
            obs_full_trajs = np.vstack(
                (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
            )


        np.random.seed(42)


        # Collect a set of trajectories from env
        for step in range(self.n_steps):
            if step % 10 == 0:
                print(f"Processed step {step} of {self.n_steps}")

            # Select action
            with torch_no_grad() as tape:
                cond = {
                "state": tf.Variable(prev_obs_venv["state"], dtype=tf.float32)
                }

                if OUTPUT_VARIABLES:
                    print("type(prev_obs_venv['state']) = ", type(prev_obs_venv["state"]))

                    print("before self.model")

                    print("self.model = ", self.model)


                    print("type(cond['state']) = ", type(cond["state"]))



                samples = self.model( cond_state = cond['state'], training=False)
                
                if OUTPUT_VARIABLES:
                    print("samples = ", samples)


                output_venv = (
                    samples.trajectories.numpy()
                )  # n_env x horizon x act



            action_venv = output_venv[:, : self.act_steps]

            # Apply multi-step action
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )

            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv

            if self.save_full_observations:  # state-only
                obs_full_venv = np.array(
                    [info["full_obs"]["state"] for info in info_venv]
                )  # n_envs x act_steps x obs_dim
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                )

            # update for next step
            prev_obs_venv = obs_venv





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


        # Plot state trajectories (only in D3IL)
        if self.traj_plotter is not None:
            self.traj_plotter(
                obs_full_trajs=obs_full_trajs,
                n_render=self.n_render,
                max_episode_steps=self.max_episode_steps,
                render_dir=self.render_dir,
                itr=0,
            )


        # Log loss and save metrics
        time = timer()
        log.info(
            f"eval: num episode {num_episode_finished:4d} | success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
        )
        np.savez(
            self.result_path,
            num_episode=num_episode_finished,
            eval_success_rate=success_rate,
            eval_episode_reward=avg_episode_reward,
            eval_best_reward=avg_best_reward,
            time=time,
        )







