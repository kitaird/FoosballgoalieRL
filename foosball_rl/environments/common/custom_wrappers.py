import logging
from collections import OrderedDict
from typing import Any, Union, Optional, Dict

import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper, Wrapper, spaces
from gymnasium.core import WrapperActType, ActType, WrapperObsType

from foosball_rl.environments.constants import WHITE_GOAL_X_POSITION, FIELD_HEIGHT


class GoalConditionedWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.original_observation_space = env.observation_space
        self.observation_space = self._make_observation_space()
        self.desired_goal = np.zeros(self.original_observation_space.shape[0])
        self.desired_goal[0:3] = np.array(
            [WHITE_GOAL_X_POSITION, 0.0, FIELD_HEIGHT])  # Where the ball should ideally be
        # The other dimensions are irrelevant as they are not used for the reward calculation

    def _make_observation_space(self) -> gym.Space:
        return spaces.Dict({
            "observation": self.original_observation_space,
            "achieved_goal": self.original_observation_space,
            "desired_goal": self.original_observation_space,
        })

    def _get_obs(self, obs: WrapperObsType):
        return OrderedDict(
            [
                ("observation", obs.copy()),
                ("achieved_goal", obs.copy()),
                ("desired_goal", self.desired_goal.copy()),
            ]
        )

    def reset(self, *, seed: int = None, options: dict[str, Any] = None) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._get_obs(obs), info

    def step(self, action: WrapperActType):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._get_obs(obs)
        reward += float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info).item())
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal: Union[int, np.ndarray], desired_goal: Union[int, np.ndarray],
                       _info: Optional[Dict[str, Any]]) -> np.float32:
        # As we are using a vectorized version, we need to keep track of the `batch_size`
        # As we are using a vectorized version, we need to keep track of the `batch_size`
        if isinstance(achieved_goal, int):
            batch_size = 1
        else:
            batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1

        reshaped_achieved_goal = np.array(achieved_goal).reshape(batch_size, -1)
        reshaped_desired_goal = np.array(desired_goal).reshape(batch_size, -1)
        achieved_ball_pos = reshaped_achieved_goal[:, 0:3]
        desired_ball_pos = reshaped_desired_goal[:, 0:3]

        distance = np.linalg.norm(achieved_ball_pos - desired_ball_pos, axis=-1)
        return -(distance > 0).astype(np.float32)


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
