import logging
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3.common.vec_env import VecVideoRecorder, VecEnv, unwrap_vec_wrapper, VecCheckNan
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from foosball_rl.environments.common.wrappers.action_space_wrappers import get_action_space_wrapper, \
    AddActionToObservationsWrapper
from foosball_rl.environments.common.wrappers.custom_vec_wrappers import VecPBRSWrapper
from foosball_rl.environments.common.wrappers.custom_wrappers import GoalEnvWrapper
from foosball_rl.environments.common.register import make_vec_env
from foosball_rl.misc.config import get_run_config
from foosball_rl.misc.utils import get_applied_gym_wrappers, get_applied_vecenv_wrappers

logger = logging.getLogger(__name__)


def create_envs(env_id: str, n_envs: int, seed: int, video_logging_path: Optional[Path], vec_normalize_path: str):
    venv = make_vec_env(env_id, n_envs, seed, wrapper_class=apply_env_wrappers)
    venv = apply_vec_env_wrappers(venv, vec_normalize_path)

    if get_run_config()['VideoRecording']['record_videos']:
        venv = enable_video_recording(venv=venv, seed=seed, video_logging_path=video_logging_path)

    log_used_wrappers(venv)
    venv.seed(seed)
    return venv


def create_eval_envs(env_id: str, n_envs: int, seed: int, video_logging_path: Optional[Path], vec_normalize_path: str = None):
    logging.info("Eval envs: Creating %s %s eval envs with seed %s", n_envs, env_id, seed)
    venv = create_envs(env_id, n_envs, seed, video_logging_path, vec_normalize_path)
    vec_normalize = unwrap_vec_wrapper(venv, VecNormalize)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return venv


def log_used_wrappers(venv: VecEnv):
    logger.info("-" * 100)
    logger.info("Gym Wrappers: %s", get_applied_gym_wrappers(venv.unwrapped.envs[0]))
    logger.info("-" * 100)
    logger.info("VecEnv Wrappers: %s", get_applied_vecenv_wrappers(venv))
    logger.info("-" * 100)


def apply_env_wrappers(env: gym.Env | gym.Wrapper):
    env = AddActionToObservationsWrapper(env)
    wrapper_conf = get_run_config()['Wrapper']
    if wrapper_conf['use_goal_env_wrapper']:
        env = GoalEnvWrapper(env)  # When using HER
    env = get_action_space_wrapper(env, wrapper_conf)
    return env


def apply_vec_env_wrappers(venv: VecEnv, vec_normalize_path: str = None):
    venv = VecPBRSWrapper(venv)
    venv = add_vec_normalize(venv, vec_normalize_path)
    venv = VecCheckNan(venv, raise_exception=True, warn_once=False)
    return venv


def add_vec_normalize(venv: VecEnv, vec_normalize_path: str) -> VecNormalize:
    if vec_normalize_path is not None:
        logger.info("Using normalized environment from %s", vec_normalize_path)
        return VecNormalize.load(venv=venv, load_path=vec_normalize_path)
    logger.info("Creating new normalized environment")
    return VecNormalize(venv=venv)


def enable_video_recording(venv: VecEnv, seed: int, video_logging_path: Optional[Path]):
    config = get_run_config()
    video_conf = config['VideoRecording']
    if video_logging_path is None:
        video_logging_path = Path(__file__).parent / config['Common']['experiment_name']
    video_length = video_conf['video_length']
    video_interval = video_conf['video_interval']
    video_log_path = video_logging_path / f"seed-{seed}" / video_conf['video_log_path_suffix']
    logger.info("VecVideoRecorder: Recording video of length %s every %s steps, saving to %s",
                video_length, video_interval, video_log_path)
    env = VecVideoRecorder(venv=venv,
                           name_prefix="rl-run-video",
                           record_video_trigger=lambda x: x % video_interval == 0,
                           video_length=video_length,
                           video_folder=video_log_path.__str__())
    return env
