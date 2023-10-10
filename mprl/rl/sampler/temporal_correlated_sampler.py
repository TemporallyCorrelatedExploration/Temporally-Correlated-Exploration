from typing import Union

import numpy as np
import torch

import mprl.util as util
from mprl.rl.critic import AbstractCritic
from mprl.rl.policy import AbstractGaussianPolicy
from mprl.util import assert_shape
from mprl.util import to_np
from mprl.util import to_ts
from mprl.util.util_learning import select_pred_pairs
from mprl.rl.sampler import BlackBoxSampler
from mprl.util import RunningMeanStd


# from stable_baselines3.common.vec_env import VecNormalize


class TemporalCorrelatedSampler(BlackBoxSampler):
    def __init__(self,
                 env_id: str,
                 num_env_train: int = 1,
                 num_env_test: int = 1,
                 episodes_per_train_env: int = 1,
                 episodes_per_test_env: int = 1,
                 discount_factor: float = 0.99,
                 disable_time_limit: bool = False,
                 use_time_feature_wrapper: bool = False,
                 sample_length: float = 1.0,
                 dt: float = 0.02,
                 dtype: str = "torch.float32",
                 device: str = "cpu",
                 seed: int = 1,
                 **kwargs):
        super().__init__(env_id,
                         num_env_train,
                         num_env_test,
                         episodes_per_train_env,
                         episodes_per_test_env,
                         discount_factor,
                         disable_time_limit,
                         use_time_feature_wrapper,
                         sample_length,
                         dt,
                         dtype,
                         device,
                         seed,
                         **kwargs)

        # Store configuration
        self.time_pairs_config = kwargs["time_pairs_config"]
        self.pred_pairs = None
        self.norm_step_obs = kwargs.get("norm_step_obs", False)
        if self.norm_step_obs:
            self.obs_rms = RunningMeanStd(name="obs_rms",
                                          shape=self.observation_space.shape,
                                          dtype=dtype, device=device)
        else:
            self.obs_rms = None
        self.norm_step_rewards = kwargs.get("norm_step_rewards", False)
        if self.norm_step_rewards:
            # TODO check shape
            self.rwd_rms = RunningMeanStd(name="rwd_rms",
                                          shape=(1,),
                                          dtype=dtype, device=device)
        else:
            self.rwd_rms = None

    def get_times(self, init_time, num_times):
        # Assert time step in an environment is the same as the MP
        assert self.debug_env.envs[0].dt == self.dt
        times = util.tensor_linspace(start=init_time + self.dt,
                                     end=init_time + num_times * self.dt,
                                     steps=num_times).T
        return times

    def get_time_pairs(self):
        pred_pairs = util.to_ts(select_pred_pairs(num_all=self.num_times,
                                                  **self.time_pairs_config),
                                torch.long, self.device)
        self.pred_pairs = pred_pairs
        return pred_pairs

    @staticmethod
    def apply_normalization(raw, rms):
        return (raw - rms.mean) / torch.sqrt(rms.var + 1e-8)

    @torch.no_grad()
    def run(self,
            training: bool,
            policy: AbstractGaussianPolicy,
            critic: AbstractCritic,
            deterministic: bool = False,
            render: bool = False,
            render_mode: str = "human",
            task_specified_metrics: list = None):
        """
        Sample trajectories

        Args:
            training: True for training, False for evaluation
            policy: policy model to get actions from
            critic: critic model to get values from
            deterministic: evaluation only, if the evaluation is deterministic
            render: evaluation only, whether render the environment
            render_mode: evaluation only, how to render the environment
            task_specified_metrics: task specific metrics

        Returns:
            rollout results
        """
        # Training or evaluation
        if training:
            assert deterministic is False and render is False
            envs = self.train_envs
            segment_init_state = envs.reset()
            num_env = self.num_env_train
            replan_per_env = \
                self.episodes_per_train_env * self.replan_per_episode
        else:
            envs = self.test_envs
            segment_init_state = envs.reset()
            if render:
                envs.render(mode=render_mode)
            num_env = self.num_env_test
            replan_per_env \
                = self.episodes_per_test_env * self.replan_per_episode

        # Determine the dimensions
        num_samples = replan_per_env * num_env
        if "num_steps" in self.task_specified_metrics:
            num_total_env_steps = 0
        else:
            num_total_env_steps = num_samples * self.num_times
        dim_obs = envs.observation_space.shape[0]
        dim_mp_params = policy.dim_out
        num_times = self.num_times
        num_dof = policy.num_dof
        # Choose prediction pairs
        pred_pairs = self.get_time_pairs()
        num_pred_pairs = pred_pairs.shape[0]

        # Storage
        list_step_states = list()
        list_step_actions = list()
        list_step_rewards = list()
        list_step_dones = list()
        list_step_values = list()
        list_segment_state = list()
        list_segment_log_prob_estimate = list()
        list_segment_init_time = list()
        list_segment_init_pos = list()
        list_segment_init_vel = list()
        list_segment_params_mean = list()
        list_segment_params_L = list()
        list_episode_reward = list()

        if self.task_specified_metrics is not None:
            dict_task_specified_metrics = dict()
            for metric in self.task_specified_metrics:
                dict_task_specified_metrics[metric] = list()

        for _ in range(replan_per_env):
            # Initial conditions
            segment_init_state = to_ts(segment_init_state,
                                       self.dtype, self.device)
            segment_init_time = segment_init_state[..., -num_dof * 2 - 1]
            segment_init_pos = segment_init_state[..., -num_dof * 2: -num_dof]
            segment_init_vel = segment_init_state[..., -num_dof:]
            assert_shape(segment_init_time, [num_env])
            assert_shape(segment_init_pos, [num_env, num_dof])
            assert_shape(segment_init_vel, [num_env, num_dof])

            segment_init_time = self.determine_segment_init_time(
                segment_init_time)

            list_segment_init_time.append(segment_init_time)
            list_segment_init_pos.append(segment_init_pos)
            list_segment_init_vel.append(segment_init_vel)

            # Policy prediction
            # Remove the desired position and velocity from observations
            segment_params_mean, segment_params_L = \
                policy.policy(segment_init_state[..., :-num_dof * 2])
            assert_shape(segment_params_mean, [num_env, dim_mp_params])
            assert_shape(segment_params_L,
                         [num_env, dim_mp_params, dim_mp_params])
            list_segment_params_mean.append(segment_params_mean)
            list_segment_params_L.append(segment_params_L)

            # Time to trajectories and log_probabilities
            step_times = self.get_times(segment_init_time, num_times)

            step_actions = policy.sample(require_grad=True,
                                         params_mean=segment_params_mean,
                                         params_L=segment_params_L,
                                         times=step_times,
                                         init_time=segment_init_time,
                                         init_pos=segment_init_pos,
                                         init_vel=segment_init_vel,
                                         use_mean=deterministic)

            segment_log_prob_estimate = \
                policy.log_prob(step_actions,
                                params_mean=segment_params_mean,
                                params_L=segment_params_L,
                                times=step_times,
                                init_time=segment_init_time,
                                init_pos=segment_init_pos,
                                init_vel=segment_init_vel,
                                pred_pairs=pred_pairs)

            assert_shape(step_actions, [num_env, num_times, num_dof * 2])
            assert_shape(segment_log_prob_estimate, [num_env, num_pred_pairs])
            list_step_actions.append(step_actions)
            list_segment_log_prob_estimate.append(segment_log_prob_estimate)

            next_segment_init_state, _, _, step_infos = \
                envs.step(to_np(step_actions))

            # Step states and values
            step_states = util.get_item_from_dicts(step_infos, "step_states")
            step_states = to_ts(np.asarray(step_states),
                                self.dtype, self.device)
            assert_shape(step_states, [num_env, num_times, dim_obs])

            # Include the initial state
            step_states = \
                torch.cat([segment_init_state[:, None], step_states], dim=-2)

            # Apply running mean std to step obs before feed it into critic
            if self.norm_step_obs:
                # Update running mean std if it is training
                if training:
                    self.obs_rms.update(step_states.view(-1, dim_obs))
                norm_step_states = self.apply_normalization(step_states,
                                                            self.obs_rms)
            else:
                norm_step_states = step_states

            # Remove the desired position and velocity from observations
            step_values = critic.critic(
                norm_step_states[..., :-num_dof * 2]).squeeze(-1)
            assert_shape(step_values, [num_env, num_times + 1])
            list_step_states.append(norm_step_states)
            list_step_values.append(step_values)

            # Update the initial state
            list_segment_state.append(segment_init_state)
            segment_init_state = next_segment_init_state

            # Actual position and velocity
            actual_pos = util.get_item_from_dicts(step_infos, "step_actual_pos")
            actual_vel = util.get_item_from_dicts(step_infos, "step_actual_vel")
            actual_pos = to_ts(np.asarray(actual_pos), self.dtype, self.device)
            actual_vel = to_ts(np.asarray(actual_vel), self.dtype, self.device)
            assert_shape(actual_pos, [num_env, num_times, num_dof])
            assert_shape(actual_vel, [num_env, num_times, num_dof])

            # Step rewards
            step_rewards = util.get_item_from_dicts(step_infos, "step_rewards")
            step_rewards = to_ts(np.asarray(step_rewards),
                                 self.dtype, self.device)
            assert_shape(step_rewards, [num_env, num_times])

            # Apply running mean std to step rewards
            if self.norm_step_rewards:
                # Update running mean std if it is training
                if training:
                    self.rwd_rms.update(step_rewards.view(-1, 1))
                norm_step_rewards = self.apply_normalization(step_rewards,
                                                             self.rwd_rms)
            else:
                norm_step_rewards = step_rewards

            list_step_rewards.append(norm_step_rewards)

            # Episode rewards
            list_episode_info = util.get_item_from_dicts(step_infos, "episode")
            episode_rewards = util.get_item_from_dicts(list_episode_info, "r")
            # All envs should be done or not done at the same time
            assert len(episode_rewards) == num_env or len(episode_rewards) == 0
            if len(episode_rewards) > 0:
                episode_rewards = to_ts(np.asarray(episode_rewards),
                                        self.dtype, self.device)
                list_episode_reward.append(episode_rewards)

            # Step dones
            step_dones = util.get_item_from_dicts(step_infos, "step_dones")
            step_dones = to_ts(np.asarray(step_dones), torch.bool, self.device)
            assert_shape(step_dones, [num_env, num_times])
            list_step_dones.append(step_dones)

            # Update training steps
            segment_length = util.get_item_from_dicts(step_infos,
                                                      "segment_length")
            self.total_training_steps += np.asarray(segment_length).sum()

            # Task specified metrics
            if self.task_specified_metrics is not None:
                for metric in self.task_specified_metrics:
                    metric_value = \
                        util.get_item_from_dicts(step_infos, metric)
                    metric_value = to_ts(np.asarray(metric_value,
                                                    dtype=np.float64)[:, -1],
                                         self.dtype, self.device)
                    dict_task_specified_metrics[metric].append(metric_value)

                    if "num_steps" == metric:
                        num_total_env_steps += metric_value.sum().item()

        # Form up return dictionary
        results = dict()
        results["step_actions"] = torch.cat(list_step_actions, dim=0)
        results["segment_log_prob_estimate"] = \
            torch.cat(list_segment_log_prob_estimate, dim=0)
        results["step_states"] = torch.cat(list_step_states, dim=0)[:, :-1]
        results["step_rewards"] = torch.cat(list_step_rewards, dim=0)
        results["segment_state"] = torch.cat(list_segment_state, dim=0)
        results["segment_reward"] = results["step_rewards"].sum(dim=-1)
        results["episode_reward"] = torch.cat(list_episode_reward, dim=0)
        results["step_dones"] = torch.cat(list_step_dones, dim=0)
        results["step_values"] = torch.cat(list_step_values, dim=0)
        results["segment_init_time"] = torch.cat(list_segment_init_time, dim=0)
        results["segment_init_pos"] = torch.cat(list_segment_init_pos, dim=0)
        results["segment_init_vel"] = torch.cat(list_segment_init_vel, dim=0)
        # Fixme, get correct time limit dones and terminal state
        results["step_time_limit_dones"] = torch.zeros([num_samples, num_times],
                                                       dtype=torch.bool,
                                                       device=self.device)
        # todo change time limit dones based on the episode rewards
        results["segment_params_mean"] = \
            torch.cat(list_segment_params_mean, dim=0)
        results["segment_params_L"] = torch.cat(list_segment_params_L, dim=0)

        if self.task_specified_metrics:
            for metric in self.task_specified_metrics:
                results[metric] = torch.cat(dict_task_specified_metrics[metric],
                                            dim=0)

        return results, num_total_env_steps

    def determine_segment_init_time(self, segment_init_time):
        """
        Determine the segment initial time based on the replan type

        Args:
            segment_init_time: original segment initial time from envs

        Returns:
            segment_init_time: new segment initial time

        """

        if self.replan_type == "sequential":
            return torch.zeros_like(segment_init_time)
        elif self.replan_type == "local":
            return segment_init_time
        else:
            raise NotImplementedError

    def save_rms(self, log_dir: str, epoch: int):
        if self.norm_step_obs:
            self.obs_rms.save(log_dir, epoch)
        if self.norm_step_rewards:
            self.rwd_rms.save(log_dir, epoch)

    def load_rms(self, log_dir: str, epoch: int):
        if self.norm_step_obs:
            self.obs_rms.load(log_dir, epoch)
        if self.norm_step_rewards:
            self.rwd_rms.load(log_dir, epoch)
