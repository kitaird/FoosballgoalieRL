from pathlib import Path
from typing import Type, Dict, Any

import gymnasium as gym
from sb3_contrib import ARS, QRDQN, TQC, TRPO, RecurrentPPO
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from foosball_rl.utils.config import get_run_config

ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    # SB3 Contrib,
    "ars": ARS,
    "qrdqn": QRDQN,
    "tqc": TQC,
    "trpo": TRPO,
    "ppo_lstm": RecurrentPPO,
}


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


LINE_SEPARATOR = '-----------------\n'


def log_experiment_config(hyperparams: Dict[str, Any], venv, save_path: Path, seed: int):
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    run_config = get_run_config()
    common_config = run_config['Common']

    with open(save_path / 'run_configuration.txt', 'w') as f:
        f.write('Run Configuration\n')
        f.write(LINE_SEPARATOR)
        f.write(f"Experiment name: {common_config['experiment_name']}\n")
        f.write(f"Environment: {common_config['env_id']}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Algorithm: {common_config['algorithm']}\n")
        f.write(LINE_SEPARATOR)
        f.write('Hyperparameters\n')
        for k, v in hyperparams.items():
            f.write(f'{k}: {v}\n')
        f.write(LINE_SEPARATOR)
        f.write('Applied wrappers\n')
        f.write(f'Gym Wrappers: {get_applied_gym_wrappers(venv.unwrapped.envs[0])}\n')
        f.write(f'VecEnv Wrappers: {get_applied_vecenv_wrappers(venv)}\n')
        f.write(LINE_SEPARATOR)
        f.write('Environment Arguments\n')
        env_cfg = venv.unwrapped.envs[0].env_config
        for k, v in env_cfg.items():
            f.write(f'{k}\n')
            for i in v.items():
                f.write(f'\t{i}\n')
        f.write(LINE_SEPARATOR)
        f.write('Run Arguments\n')
        for k, v in run_config.items():
            f.write(f'{k}\n')
            for i in v.items():
                f.write(f'\t{i}\n')
        # print_cfg(f, run_config)
        f.write(LINE_SEPARATOR)
        f.write('')
