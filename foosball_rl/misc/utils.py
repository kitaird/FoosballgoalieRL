from typing import Type, Dict

import gymnasium as gym
from sb3_contrib import ARS, QRDQN, TQC, TRPO, RecurrentPPO
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


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
