import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper, Wrapper
from gymnasium.core import WrapperActType, ActType

logger = logging.getLogger(__name__)


def get_action_space_wrapper(env: gym.Env, wrapper_conf: dict[str, Any]) -> gym.Env:
    action_space = wrapper_conf['action_space']
    if action_space == 'continuous':
        return env
    elif action_space == 'discrete':
        return DiscreteActionWrapper(env=env, lateral_bins=wrapper_conf['lateral_bins'],
                                     angular_bins=wrapper_conf['angular_bins'])
    elif action_space == 'multi-discrete':
        return MultiDiscreteActionWrapper(env=env, lateral_bins=wrapper_conf['lateral_bins'],
                                          angular_bins=wrapper_conf['angular_bins'])
    else:
        logger.error("Only \'continuous\', \'discrete\' and \'multi-discrete\' action spaces are supported"
                     "when using create_env.action_space_wrapper")
        raise ValueError(f"Unknown action space wrapper {action_space}")


class DiscreteActionWrapper(ActionWrapper):
    """
    Assumptions:
    - env has a 2D continuous action space (gym.spaces.Box)
    """

    def __init__(self, env: gym.Env, seed: int = 1, lateral_bins: int = 3, angular_bins: int = 3):
        super().__init__(env)
        self.lateral_bins = lateral_bins
        self.angular_bins = angular_bins
        self.env_action_space: gym.spaces.Box = env.action_space
        self.env_action_range = np.abs(self.env_action_space.low) + np.abs(self.env_action_space.high)
        self.lateral_increment = self.env_action_range[0] / (lateral_bins - 1)
        self.angular_increment = self.env_action_range[1] / (angular_bins - 1)
        self.action_space = gym.spaces.Discrete(lateral_bins + angular_bins, seed=seed)
        self.last_action = np.array([0, 0])

    def action(self, action: WrapperActType) -> ActType:
        current_action = self._unwrap_discrete_action(action)
        self.last_action = current_action
        return current_action

    def _unwrap_discrete_action(self, action: WrapperActType) -> ActType:
        if action < self.lateral_bins:
            return np.array([
                self.env_action_space.low[0] + (action * self.lateral_increment),
                self.last_action[1]]  # Keep the last angular action
            )
        action -= self.lateral_bins
        return np.array([
            self.last_action[0],  # Keep the last lateral action
            self.env_action_space.low[1] + (action * self.angular_increment)]
        )


class MultiDiscreteActionWrapper(ActionWrapper):

    def __init__(self, env: gym.Env, seed: int = 1, lateral_bins: int = 3, angular_bins: int = 3):
        super().__init__(env)
        self.lateral_bins = lateral_bins
        self.angular_bins = angular_bins
        self.env_action_space: gym.spaces.Box = env.action_space
        self.env_action_range = np.abs(self.env_action_space.low) + np.abs(self.env_action_space.high)
        self.lateral_increment = np.sum(self.env_action_range, axis=0) / (lateral_bins - 1)
        self.angular_increment = np.sum(self.env_action_range, axis=1) / (angular_bins - 1)
        self.action_space = gym.spaces.MultiDiscrete([self.lateral_bins, self.angular_bins], seed=seed)

    def action(self, action: WrapperActType) -> ActType:
        return self._unwrap_multi_discrete_action(action)

    def _unwrap_multi_discrete_action(self, action: WrapperActType) -> ActType:
        return np.array([
            self.env_action_space.low[0] + (action[0] * self.lateral_increment),
            self.env_action_space.low[1] + (action[1] * self.angular_increment)
        ])
