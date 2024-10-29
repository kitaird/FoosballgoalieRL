import logging
from collections import OrderedDict
from typing import Any, Optional, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import WrapperActType, WrapperObsType

from foosball_rl.environments.common.constants import WHITE_GOAL_X_POSITION


def compute_reward(achieved_goal: int | np.ndarray, desired_goal: int | np.ndarray,
                   _info: Optional[Dict[str, Any]]) -> np.float32:
    # As we are using a vectorized version, we need to keep track of the `batch_size`
    if isinstance(achieved_goal, int):
        batch_size = 1
    else:
        batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1

    reshaped_achieved_goal = np.array(achieved_goal).reshape(batch_size, -1)
    reshaped_desired_goal = np.array(desired_goal).reshape(batch_size, -1)
    achieved_ball_pos = reshaped_achieved_goal[:, 0:2]
    desired_ball_pos = reshaped_desired_goal[:, 0:2]

    return -np.linalg.norm(achieved_ball_pos - desired_ball_pos, axis=-1)


class GoalEnvWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.original_observation_space = env.observation_space
        self.observation_space = self._make_observation_space()
        self.desired_goal = np.zeros(self.original_observation_space.shape[0])
        self.desired_goal[0:2] = np.array(
            [WHITE_GOAL_X_POSITION, 0.0])  # Where the ball should ideally be from the perspective of the black team.
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
        reward += float(compute_reward(obs["achieved_goal"], obs["desired_goal"], info).item())
        return obs, reward, terminated, truncated, info

