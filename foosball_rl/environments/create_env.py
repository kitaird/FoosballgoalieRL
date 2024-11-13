from pathlib import Path
from typing import Optional
import supersuit as ss

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import unwrap_vec_wrapper
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

import logging
from foosball_rl.environments.common.register_env import make_vec_env
from foosball_rl.environments.foosball.multi_agent import env_creator
from foosball_rl.environments.foosball.multi_agent.foosball_marl import FoosballMARL
from foosball_rl.wrappers.wrapper_configuration import apply_vec_env_wrappers, apply_env_wrappers, \
    get_applied_vecenv_wrappers, get_applied_gym_wrappers

logger = logging.getLogger(__name__)


def create_marl_envs(use_team_agents: bool, n_envs: int, seed: int, video_logging_path: Optional[Path], vec_normalize_path: str) -> GymEnv:
    marl_env = ss.pettingzoo_env_to_vec_env_v1(FoosballMARL(env_creator=env_creator, use_team_agents=use_team_agents))
    marl_venv = ss.concat_vec_envs_v1(marl_env, n_envs, num_cpus=4, base_class='stable_baselines3')
    venv = apply_vec_env_wrappers(marl_venv, seed, vec_normalize_path, video_logging_path)

    # logger.info("Used Gym Wrappers: %s", get_applied_gym_wrappers(venv.unwrapped.envs[0]))
    # logger.info("Used VecEnv Wrappers: %s", get_applied_vecenv_wrappers(venv))
    venv._seeds = [seed + idx for idx in range(n_envs)]
    return venv


def create_envs(env_id: str, n_envs: int, seed: int, video_logging_path: Optional[Path], vec_normalize_path: str) -> GymEnv:
    venv = make_vec_env(env_id, n_envs, seed, wrapper_class=apply_env_wrappers)
    venv = apply_vec_env_wrappers(venv, seed, vec_normalize_path, video_logging_path)

    logger.info("Used Gym Wrappers: %s", get_applied_gym_wrappers(venv.unwrapped.envs[0]))
    logger.info("Used VecEnv Wrappers: %s", get_applied_vecenv_wrappers(venv))

    venv.seed(seed)
    return venv


def create_eval_envs(env_id: str, n_eval_envs: int, seed: int, video_logging_path: Optional[Path], vec_normalize_path: str = None) -> GymEnv:
    logging.info("Eval envs: Creating %s %s eval envs with seed %s", n_eval_envs, env_id, seed)
    venv = create_envs(env_id, n_eval_envs, seed, video_logging_path, vec_normalize_path)
    vec_normalize = unwrap_vec_wrapper(venv, VecNormalize)
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    return venv
