import logging
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv, VecCheckNan, VecNormalize, VecVideoRecorder

from foosball_rl.environments.common.wrappers.action_space_wrappers import get_action_space_wrapper
from foosball_rl.environments.common.wrappers.custom_vec_wrappers import VecPBRSWrapper
from foosball_rl.environments.common.wrappers.custom_wrappers import GoalEnvWrapper
from foosball_rl.environments.common.wrappers.observation_space_wrappers import AddActionToObservationsWrapper
from foosball_rl.utils.config import get_run_config

logger = logging.getLogger(__name__)
wrapper_conf = get_run_config()['Wrapper']
env_wrappers = wrapper_conf['EnvWrapper']
vec_env_wrappers = wrapper_conf['VecEnvWrapper']


def apply_env_wrappers(env: gym.Env | gym.Wrapper) -> gym.Env | gym.Wrapper:
    if env_wrappers['use_add_actions_to_observation_wrapper']:
        env = AddActionToObservationsWrapper(env)
    if env_wrappers['use_goal_env_wrapper']:
        env = GoalEnvWrapper(env)  # When using HER
    if env_wrappers['use_action_space_wrapper']:
        env = get_action_space_wrapper(env, env_wrappers['ActionSpaceWrapper'])
    ############################################
    # <<ExtensionPoint>>: Add more env wrapper here if needed
    ############################################
    return env


def apply_vec_env_wrappers(venv: VecEnv, seed: int, vec_normalize_path: str = None, video_logging_path: str = None) -> VecEnv:
    if vec_env_wrappers['use_vec_pbrs_wrapper']:
        venv = VecPBRSWrapper(venv)
    if vec_env_wrappers['use_vec_normalize_wrapper']:
        venv = add_vec_normalize_wrapper(venv, vec_normalize_path)
    venv = VecCheckNan(venv, raise_exception=True, warn_once=False)  # Sanity check
    if vec_env_wrappers['use_video_recording_wrapper']:
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
    return VecNormalize(venv=venv, **vec_env_wrappers['VecNormalizeWrapper'])


def add_video_recording_wrapper(venv: VecEnv, seed: int, video_logging_path: Optional[Path]) -> VecVideoRecorder:
    video_conf = vec_env_wrappers['VecVideoRecorderWrapper']
    if video_logging_path is None:
        video_logging_path = Path(__file__).parent / get_run_config()['Common']['experiment_name']
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
