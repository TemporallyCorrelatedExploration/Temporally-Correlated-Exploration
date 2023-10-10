import copy
import os.path
import sys

from typing import Union
import time
from typing import Optional
from typing import Tuple
import fancy_gym
import numpy as np
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymStepReturn

import wandb
from cw2 import cluster_work
from cw2.cw_data.cw_wandb_logger import WandBLogger
from cw2.experiment import AbstractExperiment
from cw2.experiment import AbstractIterativeExperiment

import mprl.util as util


def is_on_local_machine():
    if any(["local" in argv for argv in sys.argv]):
        return True
    else:
        return False


def is_slurm(cw: cluster_work.ClusterWork):
    if cw.args["slurm"]:
        return True
    else:
        return False


def download_saved_model(model_str: str, model_version: int):
    model_api = model_str.replace("version", f"v{model_version}")
    run = wandb.init()
    artifact = eval(model_api[11:])
    download_dir = util.make_log_dir_with_time_stamp("/tmp/download_model")
    artifact.download(root=download_dir)
    file_names = util.get_file_names_in_directory(download_dir)
    file_names.sort()
    util.print_line_title(title=f"Download model {model_version} from WandB")
    path_to_old_config = f"{download_dir}/config.yaml"
    return download_dir, path_to_old_config


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


class RLExperiment:
    def __init__(self, exp: Union[AbstractExperiment,
                                  AbstractIterativeExperiment],
                 train: bool,
                 model_str: str = None,
                 version_number: int = None,
                 epoch: int = None,
                 keep_training: bool = False):

        git_tracker = util.get_git_tracker()

        # Determine if check git status
        if util.is_debugging():
            util.print_line_title('Debug mode, do not check git repo commits.')
            # Use TkAgg backend for matplotlib to avoid error against mujoco_py
            import matplotlib
            matplotlib.use('TkAgg')
        else:
            util.print_line_title('Run mode, enforce git repo commit checking.')
            git_clean, git_status = git_tracker.check_clean_git_status(
                print_result=True)
            if not git_clean:
                assert False, "Repositories not clean"

        self.current_git_commits = git_tracker.get_git_repo_commits()

        if train:
            # Initialize experiment
            self.cw = cluster_work.ClusterWork(exp)

            # Process configs if running on local machine or slurm on cluster
            if is_slurm(self.cw) or is_on_local_machine():
                self._process_train_rep_config_file(self.cw.config)

        else:
            # Download the saved model and add to system arguments
            download_dir, path_to_old_config \
                = download_saved_model(model_str, version_number)
            sys.argv.extend([path_to_old_config, "-o", "--nocodecopy"])

            # Compare the current and the old commits
            self.old_git_commits = \
                util.parse_config(path_to_old_config)[0]["git_repos"]
            util.print_wrap_title("Git repos commits check")
            print(util.git_repos_old_vs_new(self.old_git_commits,
                                            self.current_git_commits))

            # Initialize experiment
            self.cw = cluster_work.ClusterWork(exp)

            # Process configs
            self._process_test_rep_config_file(self.cw.config.exp_configs,
                                               download_dir, epoch,
                                               keep_training)

        # Add wandb logger
        if not util.is_debugging():
            self.cw.add_logger(WandBLogger())
        self.cw.run()

    def _process_train_rep_config_file(self, config_obj):
        """
        Given processed cw2 configuration, do further process, including:
        - Overwrite log path with time stamp
        - Create model save folders
        - Overwrite random seed by the repetition number
        - Save the current repository commits
        - Make a copy of the config and restore the exp path to the original
        - Dump this copied config into yaml file into the model save folder
        - Dump the current time stamped config file in log folder to make slurm
          call bug free
        Args:
            exp_configs: list of configs processed by cw2 already

        Returns:
            None

        """
        exp_configs = config_obj.exp_configs
        formatted_time = util.get_formatted_date_time()
        # Loop over the config of each repetition
        for i, rep_config in enumerate(exp_configs):

            # Add time stamp to log directory
            log_path = rep_config["log_path"]
            rep_log_path = rep_config["_rep_log_path"]
            rep_config["log_path"] = \
                log_path.replace("log", f"log_{formatted_time}")
            rep_config["_rep_log_path"] = \
                rep_log_path.replace("log", f"log_{formatted_time}")

            # Make model save directory
            model_save_dir = util.join_path(rep_config["_rep_log_path"],
                                            "model")
            try:
                util.mkdir(os.path.abspath(model_save_dir))
            except FileExistsError:
                import logging
                logging.error(formatted_time)
                raise

            # Set random seed to the repetition number
            util.set_value_in_nest_dict(rep_config, "seed",
                                        rep_config['_rep_idx'])

            # Save repo commits
            rep_config["git_repos"] = self.current_git_commits

            # Make a hard copy of the config
            copied_rep_config = copy.deepcopy(rep_config)

            # Recover the path to its original
            copied_rep_config["path"] = copied_rep_config["_basic_path"]

            # Reset the repetition number to 1 for future test usage
            copied_rep_config["repetitions"] = 1
            if copied_rep_config.get("reps_in_parallel", False):
                del copied_rep_config["reps_in_parallel"]
            if copied_rep_config.get("reps_per_job", False):
                del copied_rep_config["reps_per_job"]

            # Delete the generated cw2 configs
            for key in rep_config.keys():
                if key[0] == "_":
                    del copied_rep_config[key]
            del copied_rep_config["log_path"]

            # Save this copied subconfig file
            util.dump_config(copied_rep_config, "config",
                             os.path.abspath(model_save_dir))

        # Save the time stamped config file in local /log directory
        time_stamped_config_path = util.make_log_dir_with_time_stamp("")
        util.mkdir(time_stamped_config_path, overwrite=True)

        config_obj.to_yaml(time_stamped_config_path,
                           relpath=False)
        config_obj.config_path = \
            util.join_path(time_stamped_config_path,
                           "relative_" + config_obj.f_name)

    @staticmethod
    def _process_test_rep_config_file(exp_configs, load_model_dir, epoch,
                                      keep_training):
        """
        Given processed cw2 configuration, do further process, including:
        - Overwrite log path with time stamp
        - Create model save folders
        - Overwrite random seed by the repetition number
        - Save the current repository commits
        - Make a copy of the config and restore the exp path to the original
        - Dump this copied config into yaml file into the model save folder

        Args:
            exp_configs: list of configs processed by cw2 already
            load_model_dir: model saved dir
            epoch: epoch of the model
            keep_training: whether to keep training
        Returns:
            None

        """
        assert len(exp_configs) == 1
        formatted_time = util.get_formatted_date_time()
        test_config = exp_configs[0]

        # Add time stamp to log directory
        log_path = test_config["log_path"]
        rep_log_path = test_config["_rep_log_path"]
        test_config["log_path"] = \
            log_path.replace("log", f"log_{formatted_time}")
        test_config["_rep_log_path"] = \
            rep_log_path.replace("log", f"log_{formatted_time}")
        test_config["load_model_dir"] = load_model_dir
        test_config["load_model_epoch"] = epoch
        test_config["repetitions"] = 1
        test_config["reps_in_parallel"] = 1
        test_config["reps_per_job"] = 1
        test_config["params"]["sampler"]["args"]["num_env_test"] = 1
        if not keep_training:
            test_config["params"]["sampler"]["args"]["num_env_train"] = 1


class TimeLimitMonitor(Monitor):
    """
    Does the same as stable_baselines3.common.monitor.Monitor but takes care of environments without TimeLimitWrapper
    todo, what is this class doing?
    """

    def __init__(self,
                 env: gym.Env,
                 filename: Optional[str] = None,
                 allow_early_resets: bool = True,
                 reset_keywords: Tuple[str, ...] = (),
                 info_keywords: Tuple[str, ...] = ()):
        super().__init__(env, filename, allow_early_resets, reset_keywords,
                         info_keywords)

        # self.max_episode_steps = env.spec.max_episode_steps  # or 1000
        # Debug
        self.max_episode_steps = 2000

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        # Monitor for done flag
        observation, reward, done, info = super().step(action)

        # Monitor when max episode steps are reached.
        # When done=True this information is already provided.
        if len(self.rewards) >= self.max_episode_steps and not done:
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len,
                       "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info

            # normally, we would require a reset on done, however for infinite horizon (i.e. no TimiLimitWrapper),
            # we just want the stats returned, after the fixed trajectory length
            # Hence, we manually reset the rewards here instead
            self.rewards = []

        return observation, reward, done, info


def make_env(env_id: str,
             seed: int,
             rank: int,
             disable_timelimit: bool = False,
             use_time_feature_wrapper: bool = False,
             wrap_monitor: bool = False) -> callable:
    """
    returns callable to create gym environment or monitor

    Args:
        env_id: gym env ID
        seed: seed for env
        rank: rank if multiple env are used
        disable_timelimit: use gym env without artificial termination signal
        use_time_feature_wrapper: Todo
        wrap_monitor: Whether to use a Monitor for episode stats or not

    Returns: callable for env constructor

    """

    assert not (disable_timelimit and use_time_feature_wrapper), \
        "Cannot disable TimeLimit and use TimeFeatureWrapper at the same time."

    def _get_env():
        env = fancy_gym.make_rank(env_id, seed=seed, rank=rank,
                                  return_callable=False)

        # Remove env from gym TimeLimitWrapper
        if disable_timelimit:
            env = env.env if isinstance(env, gym.wrappers.TimeLimit) else env
        elif use_time_feature_wrapper:
            raise NotImplementedError()

        # if log_dir is not None:
        #     import os
        #     env = gym.wrappers.Monitor(env=env, directory=os.path.join(log_dir, str(rank)))

        return TimeLimitMonitor(env) if wrap_monitor else env

    return _get_env
