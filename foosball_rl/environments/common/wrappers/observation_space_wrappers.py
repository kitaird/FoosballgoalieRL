import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gym import Wrapper
from gym.core import ActType


class AddActionToObservationsWrapper(Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.original_observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_dim = np.sum(self.action_space.shape) if env.action_space.shape != () else 1
        self.observation_space = self._extend_observation_space()
        self.logger.info("Extended observation space from %s to %s", self.original_observation_space,
                         self.observation_space)

    def _extend_observation_space(self) -> gym.Space:
        if isinstance(self.original_observation_space, gym.spaces.Box):
            low = np.concatenate((self.original_observation_space.low, -np.inf * np.ones(self.action_dim)),
                                 dtype=self.original_observation_space.dtype)
            high = np.concatenate((self.original_observation_space.high, np.inf * np.ones(self.action_dim)),
                                  dtype=self.original_observation_space.dtype)
            return gym.spaces.Box(low=low, high=high, dtype=self.original_observation_space.dtype)
        else:
            self.logger.error("Observation space %s not supported for extension", self.original_observation_space)
            raise NotImplementedError

    def reset(self, **kwargs: Any) -> Any:
        observation, info = self.env.reset(**kwargs)
        return np.concatenate([observation, np.zeros(self.action_space.shape)]), info

    def step(self, action: ActType) -> Any:
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = np.concatenate([observation, action])
        return observation, reward, terminated, truncated, info
