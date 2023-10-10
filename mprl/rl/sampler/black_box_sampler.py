from typing import Union

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

import mprl.util as util
from mprl.rl.critic import AbstractCritic
from mprl.rl.policy import AbstractGaussianPolicy
from mprl.rl.sampler.abstract_sampler import AbstractSampler
from mprl.util import assert_shape
from mprl.util import to_np
from mprl.util import to_ts


class BlackBoxSampler(AbstractSampler):
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
        super().__init__()

        # Store configuration, todo clean up the init code
        self.env_id = env_id
        self.num_env_train = self.num_cpus if num_env_train == "auto" \
            else num_env_train
        self.num_env_test = num_env_test
        self.episodes_per_train_env = episodes_per_train_env
        self.episodes_per_test_env = episodes_per_test_env
        self.discount_factor = discount_factor
        self.disable_time_limit = disable_time_limit
        self.use_time_feature_wrapper = use_time_feature_wrapper
        self.max_episode_steps = kwargs["max_episode_steps"]
        self.replan_type = kwargs.get("replan_type", "local")
        self.dt = dt
        self.num_times = int(round(sample_length / dt))
        assert self.max_episode_steps % self.num_times == 0
        self.replan_per_episode = int(
            round(self.max_episode_steps / self.num_times))
        self.dtype, self.device = util.parse_dtype_device(dtype, device)
        self.seed = seed
        self.cpu_cores = kwargs.get("cpu_cores", None)
        self.task_specified_metrics = kwargs.get("task_specified_metrics", None)
        # Get training and testing environments
        self.train_envs = self.get_env(env_type="training")
        self.test_envs = self.get_env(env_type="testing")

        # Get one more environment for debugging
        self.debug_env = self.get_env(env_type="debugging")

        # Get total training steps
        self.total_training_steps = 0

    def get_env(self, env_type: str = "training"):
        """
        Get training and testing environments

        Args:
            env_type: flag of training, testing or debugging

        Returns:
            training or testing environments
        """

        if env_type == "training":
            num_env = self.num_env_train
            seed = self.seed
        elif env_type == "testing":
            num_env = self.num_env_test
            seed = self.seed + 10000
        elif env_type == "debugging":
            num_env = 1
            seed = self.seed + 20000
        else:
            raise ValueError("Unknown env_type: {}".format(env_type))

        VecEnv = SubprocVecEnv if num_env > 1 else DummyVecEnv
        env_fns = [util.make_env(env_id=self.env_id, seed=seed, rank=i,
                                 use_time_feature_wrapper=
                                 self.use_time_feature_wrapper,
                                 disable_timelimit=self.disable_time_limit,
                                 wrap_monitor=True) for i in range(num_env)]

        envs = VecEnv(env_fns)

        if self.cpu_cores is not None and num_env > 1:
            assert len(self.cpu_cores) >= num_env, \
                "The number of cpu cores should be greater or equal to num of " \
                "environment."
            cores_per_env = int(len(self.cpu_cores) / num_env)
            cpu_cores_list = list(self.cpu_cores)
            env_pids = [envs.processes[i].pid for i in range(num_env)]
            for i, pid in enumerate(env_pids):
                cores_env = cpu_cores_list[i * cores_per_env:
                                           (i + 1) * cores_per_env]
                util.assign_process_to_cpu(pid, set(cores_env))

        return envs

    @torch.no_grad()
    def run(self,
            training: bool,
            policy: AbstractGaussianPolicy,
            critic: AbstractCritic,
            deterministic: bool = False,
            render: bool = False,
            render_mode: str = "human",
            task_specified_metrics: list = None): # fixme, not used
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

        # Storage
        list_segment_state = list()
        list_segment_action = list()
        list_segment_log_prob = list()
        list_segment_reward = list()
        list_segment_done = list()
        list_segment_value = list()
        list_segment_params_mean = list()
        list_segment_params_L = list()
        list_episode_reward = list()

        if self.task_specified_metrics is not None:
            dict_task_specified_metrics = dict()
            for metric in self.task_specified_metrics:
                dict_task_specified_metrics[metric] = list()

        for _ in range(replan_per_env):
            segment_init_state = to_ts(segment_init_state,
                                       self.dtype, self.device)
            # Policy prediction
            segment_params_mean, segment_params_L = \
                policy.policy(segment_init_state)
            assert_shape(segment_params_mean, [num_env, dim_mp_params])
            assert_shape(segment_params_L,
                         [num_env, dim_mp_params, dim_mp_params])
            list_segment_params_mean.append(segment_params_mean)
            list_segment_params_L.append(segment_params_L)

            segment_action = policy.sample(require_grad=True,
                                           params_mean=segment_params_mean,
                                           params_L=segment_params_L,
                                           use_mean=deterministic)
            # print(f"time:", segment_action[0, :2])
            # print(f"w: {segment_action[0, 2:].reshape([-1, 3])}")

            segment_log_prob = policy.log_prob(segment_action,
                                               params_mean=segment_params_mean,
                                               params_L=segment_params_L)

            assert_shape(segment_action,
                         [num_env, *self.debug_env.action_space.shape])
            assert_shape(segment_log_prob, [num_env])
            list_segment_action.append(segment_action)
            list_segment_log_prob.append(segment_log_prob)

            # Values
            segment_init_state = to_ts(segment_init_state,
                                       self.dtype, self.device)
            values = critic.critic(segment_init_state).squeeze(-1)
            assert_shape(segment_init_state, [num_env, dim_obs])
            assert_shape(values, [num_env])
            list_segment_state.append(segment_init_state)
            list_segment_value.append(values)

            segment_init_state, segment_reward, segment_done, segment_info = \
                envs.step(to_np(segment_action))

            # Segment rewards
            assert_shape(segment_reward, [num_env])
            segment_reward = to_ts(np.asarray(segment_reward),
                                   self.dtype, self.device)
            list_segment_reward.append(segment_reward)

            # Episode rewards
            list_episode_info = \
                util.get_item_from_dicts(segment_info, "episode")
            episode_reward = util.get_item_from_dicts(list_episode_info, "r")
            # All envs should be done or not done at the same time
            assert len(episode_reward) == num_env or len(episode_reward) == 0
            if len(episode_reward) > 0:
                episode_reward = to_ts(np.asarray(episode_reward),
                                       self.dtype, self.device)
                list_episode_reward.append(episode_reward)

            # Step dones
            segment_done = to_ts(np.asarray(segment_done), torch.bool,
                                 self.device)
            assert_shape(segment_done, [num_env])
            list_segment_done.append(segment_done)

            # Update training steps
            segment_lengths = util.get_item_from_dicts(segment_info,
                                                       "trajectory_length")
            self.total_training_steps += np.asarray(segment_lengths).sum()

            # Task specified metrics
            if self.task_specified_metrics is not None:
                for metric in self.task_specified_metrics:
                    metric_value = \
                        util.get_item_from_dicts(segment_info, metric,
                                                 lambda x: x[-1])

                    ep_metric_value = to_ts(metric_value,
                                            self.dtype, self.device)
                    dict_task_specified_metrics[metric].append(ep_metric_value)
                    if "num_steps" == metric:
                        num_total_env_steps += ep_metric_value.sum().item()

        # Form up return dictionary
        results = dict()
        results["segment_action"] = torch.cat(list_segment_action, dim=0)
        results["segment_log_prob"] = torch.cat(list_segment_log_prob, dim=0)
        results["segment_state"] = torch.cat(list_segment_state, dim=0)
        results["segment_reward"] = torch.cat(list_segment_reward, dim=0)
        results["episode_reward"] = torch.cat(list_episode_reward, dim=0)

        results["segment_done"] = torch.cat(list_segment_done, dim=0)
        results["segment_value"] = torch.cat(list_segment_value, dim=0)
        results["segment_params_mean"] = torch.cat(list_segment_params_mean,
                                                   dim=0)
        results["segment_params_L"] = torch.cat(list_segment_params_L, dim=0)

        if self.task_specified_metrics:
            for metric in dict_task_specified_metrics:
                results[metric] = torch.cat(dict_task_specified_metrics[metric],
                                            dim=0)

        return results, num_total_env_steps
