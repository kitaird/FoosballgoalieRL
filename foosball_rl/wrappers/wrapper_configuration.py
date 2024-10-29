import logging
from pathlib import Path
from typing import Optional

import gymnasium as gym
import yaml
from stable_baselines3.common.vec_env import VecEnv, VecCheckNan, VecNormalize, VecVideoRecorder, VecEnvWrapper

from foosball_rl import EXPERIMENT_NAME
from foosball_rl.environments.common.wrappers.action_space_wrappers import get_action_space_wrapper
from foosball_rl.environments.common.wrappers.custom_vec_wrappers import VecPBRSWrapper
from foosball_rl.environments.common.wrappers.custom_wrappers import GoalEnvWrapper
from foosball_rl.environments.common.wrappers.observation_space_wrappers import AddActionToObservationsWrapper

logger = logging.getLogger(__name__)

wrapper_config_path = Path(__file__).parent / 'wrapper_config.yml'

with open(wrapper_config_path) as f:
    wrapper_conf = yaml.safe_load(f)

ENV_WRAPPERS = wrapper_conf['EnvWrapper']
VEC_ENV_WRAPPERS = wrapper_conf['VecEnvWrapper']


def apply_env_wrappers(env: gym.Env | gym.Wrapper) -> gym.Env | gym.Wrapper:
    if ENV_WRAPPERS['use_add_actions_to_observation_wrapper']:
        env = AddActionToObservationsWrapper(env)
    if ENV_WRAPPERS['use_goal_env_wrapper']:
        env = GoalEnvWrapper(env)  # When using HER
    if ENV_WRAPPERS['use_action_space_wrapper']:
        env = get_action_space_wrapper(env, ENV_WRAPPERS['ActionSpaceWrapper'])
    ############################################
    # <<ExtensionPoint>>: Add more env wrapper here if needed
    ############################################
    return env


def apply_vec_env_wrappers(venv: VecEnv, seed: int, vec_normalize_path: str = None, video_logging_path: str = None) -> VecEnv:
    if VEC_ENV_WRAPPERS['use_vec_pbrs_wrapper']:
        venv = VecPBRSWrapper(venv)
    if VEC_ENV_WRAPPERS['use_vec_normalize_wrapper']:
        venv = add_vec_normalize_wrapper(venv, vec_normalize_path)
    venv = VecCheckNan(venv, raise_exception=True, warn_once=False)  # Sanity check
    if VEC_ENV_WRAPPERS['use_video_recording_wrapper']:
        venv = add_video_recording_wrapper(venv=venv, seed=seed, video_logging_path=video_logging_path)
    ############################################
    # <<ExtensionPoint>>: Add more vec env wrapper here if needed
    ############################################
    return venv


def add_vec_normalize_wrapper(venv: VecEnv, vec_normalize_path: str) -> VecNormalize:
    if vec_normalize_path is not None:
        logger.info("Loading normalized environment from %s", vec_normalize_path)
        return VecNormalize.load(venv=venv, load_path=vec_normalize_path)
    logger.info("Creating new VecNormalizeWrapper for environment")
    return VecNormalize(venv=venv, **VEC_ENV_WRAPPERS['VecNormalizeWrapper'])


def add_video_recording_wrapper(venv: VecEnv, seed: int, video_logging_path: Optional[Path]) -> VecVideoRecorder:
    video_conf = VEC_ENV_WRAPPERS['VecVideoRecorderWrapper']
    if video_logging_path is None:
        video_logging_path = Path(__file__).parent / EXPERIMENT_NAME
    video_length = video_conf['video_length']
    video_interval = video_conf['video_interval']
    video_log_path = video_logging_path / f"seed-{seed}" / video_conf['video_log_path_suffix']
    logger.info("VecVideoRecordingWrapper: Recording video of length %s every %s steps, saving to %s",
                video_length, video_interval, video_log_path)
    env = VecVideoRecorder(venv=venv,
                           name_prefix="rl-run-video",
                           record_video_trigger=lambda x: x % video_interval == 0,
                           video_length=video_length,
                           video_folder=video_log_path.__str__())
    return env


def get_applied_gym_wrappers(env: gym.Env):
    env_tmp = env
    wrappers = []
    while isinstance(env_tmp, gym.Wrapper):
        wrappers.append(env_tmp.__class__.__name__)
        env_tmp = env_tmp.env
    return wrappers


def get_applied_vecenv_wrappers(venv: VecEnv):
    venv_tmp = venv
    wrappers = []
    while isinstance(venv_tmp, VecEnvWrapper):
        wrappers.append(venv_tmp.__class__.__name__)
        venv_tmp = venv_tmp.venv
    return wrappers
