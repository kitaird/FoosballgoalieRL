# noqa: D212, D415
"""
Kicker environment for reinforcement learning.

Dimensions of the playing field:
- Long side of the kicker (x-axis) : [-0.67, 0.67]
- Short side of the kicker (y-axis): [-0.355, 0.355]
- Height of the kicker (z-axis): [0.0, 0.9]

Meaning of mujoco sensors:
-     0: white goal sensor
-     1: black goal sensor
-   2-4: ball velocity sensor (x,y,z)
-   5-7: ball accelerator sensor (x,y,z)
-     8: black goalie lateral position sensor
-     9: black goalie lateral velocity sensor
-    10: black goalie angular position sensor
-    11: black goalie angular velocity sensor
- 12-15: black defense
- 16-19: black midfield
- 20-23: black striker
- 24-39: white team

Meaning of mujoco actuators:
-    0: black goalie lateral actuator
-    1: black goalie angular actuator
-  2-3: black defense
-  4-5: black midfield
-  6-7: black striker
- 8-15: white team

Meaning of qpos (positions)
-     0: ball position x
-     1: ball position y
-     2: ball position z
-     3: ball quaternion w
-     4: ball quaternion x
-     5: ball quaternion y
-     6: ball quaternion z
-     7: black goalie lateral
-     8: black goalie angular
-  9-10: black defense
- 11-12: black midfield
- 13-14: black striker
- 15-22: white team

Meaning of qvel (velocities):
-     0: ball lateral x
-     1: ball lateral y
-     2: ball lateral z
-     3: ball angular x
-     4: ball angular y
-     5: ball angular z
-     6: black goalie lateral
-     7: black goalie angular
-   8-9: black defense
- 10-11: black midfield
- 12-13: black striker
- 14-21: white team
"""
import logging
from pathlib import Path
from typing import Dict, Any

import gymnasium as gym
import mujoco
import numpy as np

from foosball_rl.environments.common.mujoco_viewer import MujocoViewer
from foosball_rl.environments.common.constraints import ball_in_black_goal_bounds, \
    ball_in_white_goal_bounds
from foosball_rl.environments.foosball.single_agent.episode_definition import EpisodeDefinition, \
    FoosballEpisodeDefinition


class RawEnv(gym.Env):

    metadata = {
        "name": "Foosball_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }
    reward_range = (-1, 1)
    camera_id = "table_view"

    def __init__(self,
                 step_frequency: int = 16,
                 render_mode: str = None,
                 use_image_obs: bool = False,
                 episode_definition: EpisodeDefinition = None,
                 env_config: Dict[str, Any] = None):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._env_config = env_config
        self.render_mode = render_mode
        self.use_image_obs = use_image_obs
        if self.use_image_obs:
            assert self.render_mode == "rgb_array", "Image observations are only supported with render_mode='rgb_array'"
        self.episode_definition = episode_definition if episode_definition is not None else FoosballEpisodeDefinition()

        xml_path = (Path(__file__).parent.parent / 'foosball.xml').as_posix()
        self._mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_data: mujoco.MjData = mujoco.MjData(self.mj_model)
        self.episode_definition.mj_data = self.mj_data

        self.nr_substeps = 1
        self.nr_intermediate_steps = step_frequency
        self.dt = self.mj_model.opt.timestep * self.nr_substeps * self.nr_intermediate_steps

        self.renderer = self._initialize_renderer()

        self.action_space = self._initialize_action_space()
        self.observation_space = self._initialize_observation_space()

        self._log_initialization()

    def _log_initialization(self):
        self.logger.info("Initialized Goalkeeper Environment with episode definition: %s", self.episode_definition)
        self.logger.info("Using Observation Space: %s", self.observation_space)
        self.logger.info("Using Action Space: %s", self.action_space)

    def _initialize_renderer(self):
        if self.render_mode == "human":
            return MujocoViewer(self.mj_model, self.mj_data, self.dt)
        elif self.render_mode == "rgb_array":
            return mujoco.Renderer(self.mj_model, height=480, width=640)

    def _initialize_action_space(self, seed=None):
        action_bounds = self.mj_model.actuator_ctrlrange.copy().astype(np.float32)
        action_low, action_high = action_bounds.T
        return gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32, seed=seed)

    def _initialize_observation_space(self):
        if self.use_image_obs:
            return gym.spaces.Box(low=0, high=255, shape=self.render().shape, dtype=np.uint8)
        else:
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._feature_vector_obs().shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.logger.info("Setting seed to %s", seed)
            self._initialize_action_space(seed=seed)
            self.episode_definition.seed(seed=seed)
            super().reset(seed=seed)

        self.episode_definition.initialize_episode()

        mujoco.mj_forward(self.mj_model, self.mj_data)

        if self.render_mode == "human":
            self.render()

        return self.get_observation(), {}

    def step(self, actions) -> tuple[gym.core.ObsType, float, bool, bool, dict[str, Any]]:
        mj_ctrl = np.concatenate([np.array(a) for a in actions.values()]).reshape(16)
        for _ in range(self.nr_intermediate_steps):
            self.mj_data.ctrl = mj_ctrl
            mujoco.mj_step(self.mj_model, self.mj_data, self.nr_substeps)

        next_state = self.get_observation()
        reward, reward_info = self._get_reward()

        terminated = self.episode_definition.is_terminated()
        truncated = self.episode_definition.is_truncated()

        info = {**reward_info}

        if terminated or truncated:
            info["terminal_observation"] = next_state

        if self.render_mode == "human":
            self.render()

        return next_state, reward, terminated, truncated, info

    def get_observation(self):
        if self.use_image_obs:
            return self.render()
        else:
            return self._feature_vector_obs()

    def _get_reward(self):
        ball_pos = self.mj_data.qpos[0:3].copy()

        black_conceded = self.mj_data.sensordata[0].copy() > 0 or ball_in_black_goal_bounds(ball_pos)
        white_conceded = self.mj_data.sensordata[1].copy() > 0 or ball_in_white_goal_bounds(ball_pos)

        assert not (black_conceded and white_conceded)

        reward = 0

        if black_conceded:
            reward = -1
        elif white_conceded:
            reward = 1

        info = {
            "black_conceded": black_conceded,
            "white_conceded": white_conceded
        }

        return reward, info

    def render(self):
        if self.render_mode == "human":
            self.renderer.render()
            return None
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.mj_data, self.camera_id)
            return self.renderer.render().copy()

    def _feature_vector_obs(self):
        sensors_wo_goals = self.mj_data.sensordata[2:38]

        return np.concatenate([
            self.mj_data.qpos[0:3].copy(),
            sensors_wo_goals.copy()
        ]).astype(np.float32)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mj_data(self) -> mujoco.MjData:
        return self._mj_data

    @mj_data.setter
    def mj_data(self, value) -> None:
        self._mj_data = value

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config
