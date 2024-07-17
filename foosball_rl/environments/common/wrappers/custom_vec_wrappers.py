from typing import Callable, Dict

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs
from foosball_rl.environments.constants import WHITE_GOAL_X_POSITION

WHITE_GOAL_X_Y_COORDINATES = np.array([WHITE_GOAL_X_POSITION, 0])


def euclidean_distance(obs: np.ndarray) -> np.ndarray:
    ball_positions = obs[:, 0:2]
    goal_position = np.ones(ball_positions.shape, dtype=np.float32) * WHITE_GOAL_X_Y_COORDINATES
    return -np.linalg.norm(ball_positions-goal_position, axis=-1)


WEIGHTING_FACTOR = np.array([0.8, 0.2])


def weighted_stepwise_function(obs: np.ndarray) -> np.ndarray:
    ball_positions = obs[:, 0:2]
    x_potentials = ball_positions[:, 0] * 0.8217 + 0.5  # x: [-0.6085 , 0.6085], 0 ≤ x_pot ≤ 1
    y_potentials = np.abs(ball_positions[:, 1]) * -2.08 + 0.71  # y: [-0.34 , 0.34], symmetrical around 0, 0 ≤ y_pot ≤ 1
    potentials = np.column_stack([x_potentials, y_potentials])
    clipped_potentials = np.clip(potentials, 0, 1)
    return np.dot(clipped_potentials, WEIGHTING_FACTOR)


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
                 potential_f: Callable[[np.ndarray], np.ndarray] = euclidean_distance,
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
        obs, rewards, dones, infos = self.venv.step_wait()
        current_potentials = self.potential(obs) if not isinstance(obs, Dict) else self.potential(obs['observation'])
        potential_differences = self.gamma * current_potentials - self.last_potentials
        self.last_potentials = current_potentials
        rewards += potential_differences
        return obs, rewards, dones, infos
