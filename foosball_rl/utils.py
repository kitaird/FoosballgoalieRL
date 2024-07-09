import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


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
