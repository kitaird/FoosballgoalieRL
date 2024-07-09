import logging
from configparser import ConfigParser
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, VecVideoRecorder, VecEnv)
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from foosball_rl.environments.common.custom_vec_wrappers import VecPBRSWrapper
from foosball_rl.environments.common.custom_wrappers import DiscreteActionWrapper, MultiDiscreteActionWrapper, \
    AddActionToObservationsWrapper, GoalConditionedWrapper
from foosball_rl.utils import get_applied_gym_wrappers, get_applied_vecenv_wrappers

logger = logging.getLogger(__name__)


def create_env(env_id: str, config: ConfigParser, seed: int, video_logging_path: Optional[Path], vec_normalize_path: str = None):
    env = gym.make(env_id)
    env = apply_wrappers(config, env, vec_normalize_path)
    env.seed(seed)
    if config['VideoRecording'].getboolean('record_videos'):
        env = enable_video_recording(env, config, seed, video_logging_path)

    logger.info("-" * 50)
    logger.info("VecEnv Wrappers: %s", get_applied_vecenv_wrappers(env))
    logger.info("-" * 50)
    return env


def apply_wrappers(config: ConfigParser, env: gym.Env, vec_normalize_path: str = None):
    env = AddActionToObservationsWrapper(env)
    env = GoalConditionedWrapper(env)  # When using HER
    env = action_space_wrapper(env, config)
    env = Monitor(env)
    venv = DummyVecEnv([lambda: env])  # VecEnv from here on
    venv = VecPBRSWrapper(venv=venv, gamma=config['Algorithm'].getfloat('discount_factor'))
    if vec_normalize_path is not None:
        logger.info("Using normalized environment from %s", vec_normalize_path)
        venv = VecNormalize.load(venv=venv, load_path=vec_normalize_path)
    else:
        logger.info("Creating new normalized environment")
        venv = VecNormalize(venv=venv, gamma=config['Algorithm'].getfloat('discount_factor'))

    logger.info("-" * 50)
    logger.info("Gym Wrappers: %s", get_applied_gym_wrappers(env))
    logger.info("-" * 50)
    return venv


def action_space_wrapper(env: gym.Env, config: ConfigParser):
    wrapper_conf = config['Wrapper']
    action_space = wrapper_conf['action_space']
    if action_space == 'continuous':
        return env  # Already continuous
    elif action_space == 'discrete':
        return DiscreteActionWrapper(env=env, lateral_bins=wrapper_conf.getint('lateral_bins'),
                                     angular_bins=wrapper_conf.getint('angular_bins'))
    elif action_space == 'multi-discrete':
        return MultiDiscreteActionWrapper(env=env, lateral_bins=wrapper_conf.getint('lateral_bins'),
                                          angular_bins=wrapper_conf.getint('angular_bins'))
    else:
        logger.error("Only \'continuous\', \'discrete\' and \'multi-discrete\' action spaces are supportted"
                     "when using create_env.action_space_wrapper")
        raise ValueError(f"Unknown action space wrapper {action_space}")


def enable_video_recording(venv: VecEnv, config: ConfigParser, seed: int, video_logging_path: Optional[Path]):
    video_conf = config['VideoRecording']
    if video_logging_path is None:
        video_logging_path = Path(__file__).parent / config['Common']['experiment_name']
    video_log_path = video_logging_path / video_conf['video_log_path_suffix']
    video_interval = video_conf.getint('video_interval')
    video_length = video_conf.getint('video_length')
    logger.info("VecVideoRecorder: Recording video every %s steps with a length of %s frames, saving to %s",
                video_interval, video_length, video_log_path)
    env = VecVideoRecorder(venv=venv, name_prefix=f"rl-run-video-seed-{seed}",
                           record_video_trigger=lambda x: x % video_interval == 0,
                           video_length=video_length,
                           video_folder=video_log_path.__str__())
    return env
