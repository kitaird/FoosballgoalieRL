from typing import Callable, Dict

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs

WEIGHTING_FACTOR = np.array([0.8, 0.2])


def potential_function(obs: np.ndarray) -> np.ndarray:
    if isinstance(obs, Dict):
        obs = obs['observation']  # Handling GoalConditionedWrapper
    ball_positions = obs[:, 0:2]
    potentials = np.column_stack([
        ball_x_potential(ball_positions[:, 0]),
        ball_y_potential(ball_positions[:, 1])
    ])
    clipped_potentials = np.clip(potentials, 0, 1)
    return np.dot(clipped_potentials, WEIGHTING_FACTOR)


def ball_x_potential(x):
    # x: [-0.6085 , 0.6085]
    # 0 ≤ x_pot ≤ 1
    return x * 0.8217 + 0.5


def ball_y_potential(y):
    # y: [-0.34 , 0.34]
    # symmetrical around 0
    # 0 ≤ y_pot ≤ 1
    return np.abs(y) * -2.08 + 0.71


class VecPBRSWrapper(VecEnvWrapper):
    """
    Potential-based reward shaping wrapper for vectorized environments, based on:
    Ng, A. Y., Harada, D., & Russell, S. (1999, June).
    Policy invariance under reward transformations: Theory and application to reward shaping.
    In Icml (Vol. 99, pp. 278-287).

    The potential function is a weighted sum of the x and y coordinates of the ball position.

    WARNING: Works currently only with feature-based observations, not images.
    """
    def __init__(self, venv: VecEnv,
                 potential_f: Callable[[np.ndarray], np.ndarray] = potential_function,
                 gamma: float = 0.99):
        VecEnvWrapper.__init__(self, venv, venv.observation_space, venv.action_space)
        self.venv = venv
        self.gamma = gamma
        self.last_potentials = None
        self.potential = potential_f

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.last_potentials = np.zeros(self.num_envs)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        current_potentials = self.potential(observations)
        potential_differences = self.gamma * current_potentials - self.last_potentials
        self.last_potentials = current_potentials
        rewards += potential_differences
        return observations, rewards, dones, infos
