from pathlib import Path
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec

from foosball_rl.kicker.viewer import MujocoViewer


class Kicker(gym.Env):
    """
    Kicker environment for reinforcement learning.

    Dimensions of the playing field:
    - Long side of the kicker (x-axis) : [-1.34, 1.34]
    - Short side of the kicker (y-axis): [-0.71, 0.71]
    - Height of the kicker (z-axis): [0.0, 0.9]

    Meaning of mujoco sensors:
    - 0: white goal sensor
    - 1: black goal sensor
    - 2-4: ball velocity sensor
    - 5-7: ball accelerator sensor
    - 8: black goalie lateral position sensor
    - 9: black goalie lateral velocity sensor
    - 10: black goalie angular position sensor
    - 11: black goalie angular velocity sensor

    Meaning of mujoco actuators:
    - 0: black goalie lateral actuator
    - 1: black goalie angular actuator

    Meaning of qpos (positions)
    - 0: black goalie lateral
    - 1: black goalie angular
    - 2: ball x
    - 3: ball y
    - 4: ball z
    - 5: ball quaternion w
    - 6: ball quaternion x
    - 7: ball quaternion y
    - 8: ball quaternion z

    Meaning of qvel (velocities):
    - 0: black goalie lateral
    - 1: black goalie angular
    - 2: ball x
    - 3: ball y
    - 4: ball z
    - 5: ball angular x
    - 6: ball angular y
    - 7: ball angular z

    """
    spec = EnvSpec("KickerEnv", "no-entry-point")

    def __init__(self,
                 seed: int = 10,
                 horizon: int = 1000,
                 continuous_act_space: bool = True,
                 multi_discrete_act_space: bool = False,
                 image_obs_space: bool = False,
                 end_episode_on_struck_goal: bool = True,
                 end_episode_on_conceded_goal: bool = True,
                 reset_goalie_position: bool = True,
                 render_training: bool = True,
                 lateral_bins: int = 4,
                 angular_bins: int = 8,
                 step_frequency: int = 16):
        super().__init__()
        self.seed = seed
        self.random_num_generator = np.random.default_rng(seed=self.seed)
        self.horizon = horizon
        self.continuous_act_space = continuous_act_space
        self.multi_discrete_act_space = multi_discrete_act_space
        self.image_obs_space = image_obs_space
        self.end_episode_on_struck_goal = end_episode_on_struck_goal
        self.end_episode_on_conceded_goal = end_episode_on_conceded_goal
        self.reset_goalie_position = reset_goalie_position

        # Discrete action space parameters
        self.lateral_bins: int = lateral_bins
        self.angular_bins: int = angular_bins
        self.lateral_increment = 2 / (self.lateral_bins - 1)
        self.angular_increment = 2 / (self.angular_bins - 1)

        # Mujoco parameters
        xml_path = (Path(__file__).resolve().parent / "data" / "kicker.xml").as_posix()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.nr_substeps = 1
        self.nr_intermediate_steps = step_frequency
        self.dt = self.model.opt.timestep * self.nr_substeps * self.nr_intermediate_steps

        self.camera_id = "table_view"
        self.render_mode = "human" if render_training else "rgb_array"

        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.viewer = MujocoViewer(self.model, self.dt)

        # General reinforcement learning attributes
        self._initialize_last_action()
        self.action_space = self._make_action_space()
        self.observation_space = self._make_observation_space()
        self.episode_step = 0
        print("-" * 50)
        print("Using Observation Space: ", self.observation_space)
        print("Using Action Space: ", self.action_space)
        print("-" * 50)

    def set_render_mode(self, render_mode: str):
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        if seed is not None and seed != self.seed:
            self.seed = seed
            self.random_num_generator = np.random.default_rng(seed=self.seed)

        self.episode_step = 0
        self._initialize_last_action()

        if self.reset_goalie_position:
            qpos = np.zeros(self.model.nq)
        else:
            qpos = self.data.qpos.copy()

        # Ball Position
        qpos[2] = -0.4479 + 0.075  # x ball
        qpos[3] = self.random_num_generator.uniform(low=-0.3, high=0.3)  # y ball
        qpos[4] = 0.615  # z ball

        if self.reset_goalie_position:
            qvel = np.zeros(self.model.nv)
        else:
            qvel = self.data.qvel.copy()

        # Ball Velocity
        goal_height = -1.25
        vels = calculate_axis_velocities(qpos[2], qpos[3], goal_height, [-0.175, 0.175], 2, self.random_num_generator)
        qvel[2] = vels[0]
        qvel[3] = vels[1]

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        if self.viewer and self.render_mode == "human":
            self.viewer.render(self.data)

        return self.get_observation(), {}

    def step(self, action):
        for _ in range(self.nr_intermediate_steps):
            if self.continuous_act_space:
                self.data.ctrl = action
            else:
                self.data.ctrl = self._unwrap_discrete_action(action)
            mujoco.mj_step(self.model, self.data, self.nr_substeps)

        if self.viewer and self.render_mode == "human":
            self.viewer.render(self.data)

        self._update_last_action(action)
        self.episode_step += 1

        next_state = self.get_observation()
        reward, r_info = self.get_reward()

        terminated = r_info["black_conceded"]

        if self.end_episode_on_struck_goal:
            terminated = terminated or r_info["white_conceded"]
        if self.end_episode_on_conceded_goal:
            terminated = terminated or r_info["black_conceded"]

        ball_data = r_info["ball_position"]
        ball_outside_bounds = is_ball_outside_bounds(ball_data)
        ball_stopped_outside_goalie_space = self._ball_stopped() and is_ball_outside_goalie_space(ball_data)
        truncated = self.episode_step >= self.horizon or ball_outside_bounds or ball_stopped_outside_goalie_space

        info = {**r_info, "outside_bounds": 1 if ball_outside_bounds else 0}

        if ball_outside_bounds:
            info["ball_outside_bounds_x"] = ball_data[0]
            info["ball_outside_bounds_y"] = ball_data[1]
            info["ball_outside_bounds_z"] = ball_data[2]
        if truncated or terminated:
            info["final_observation"] = next_state

        return next_state, reward, terminated, truncated, info

    def get_observation(self):
        if self.image_obs_space:
            return self._get_image_observation()
        else:
            return self._get_discrete_observation()

    def get_reward(self):
        ball_pos = self.data.body("ball").xpos

        black_conceded = self.data.sensor("black_goal_sensor").data[0] > 0 or is_ball_in_black_goal_bounds(ball_pos)
        white_conceded = self.data.sensor("white_goal_sensor").data[0] > 0 or is_ball_in_white_goal_bounds(ball_pos)

        reward = 0

        assert not (black_conceded and white_conceded)

        if black_conceded:
            reward = -1
        elif white_conceded:
            reward = 1

        info = {
            "black_conceded": 1 if black_conceded else 0,
            "white_conceded": 1 if white_conceded else 0,
            "ball_position": ball_pos
        }

        return reward, info

    def render(self, mode=None):
        if mode == "rgb_array" or self.render_mode == "rgb_array":
            return self._get_image_observation()
        if self.viewer and mode == "human":
            self.viewer.render(self.data)

    def close(self):
        if self.viewer:
            self.viewer.close()

    #### Helper Functions ####

    def _unwrap_discrete_action(self, action) -> np.ndarray:
        if self.multi_discrete_act_space:
            return self._flatten_multi_discrete_action(action)
        else:
            return self._flatten_discrete_action(action)

    def _flatten_multi_discrete_action(self, action):
        return np.array([-1 + (self.lateral_increment * action[0]), -1 + (self.angular_increment * action[1])])

    def _flatten_discrete_action(self, action):
        if action < self.lateral_bins:
            return np.array([-1 + (action * self.lateral_increment), self.last_action[1]])
        action -= self.lateral_bins
        return np.array([self.last_action[0], -1 + (action * self.angular_increment)])

    def _get_discrete_observation(self) -> np.ndarray:
        sensors_wo_goals = self.data.sensordata[2:12]

        return np.concatenate([
            sensors_wo_goals,
            self.data.body("ball").xpos,
            self._get_last_action()
        ])

    def _initialize_last_action(self) -> None:
        self.last_action = np.zeros(2)
        self.last_discrete_action = 0

    def _get_last_action(self) -> np.ndarray:
        return np.array([self.last_discrete_action]) if isinstance(self.action_space, gym.spaces.Discrete) else self.last_action

    def _update_last_action(self, action) -> None:
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.last_discrete_action = action
        else:
            self.last_action = action

    def _get_image_observation(self) -> np.ndarray:
        self.renderer.update_scene(self.data, self.camera_id)
        return self.renderer.render()

    def _ball_stopped(self):
        ball_x_vel = self.data.qvel[2]
        ball_z_vel = self.data.qvel[3]
        return np.abs(ball_z_vel) < 0.01 and np.abs(ball_x_vel) < 0.01

    def _make_action_space(self) -> spaces.Space:
        """
        Helper to create action space

        :return: the environment action space
        """
        if self.continuous_act_space:
            action_bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
            action_low, action_high = action_bounds.T
            return gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32, seed=self.seed)
        elif self.multi_discrete_act_space:
            return gym.spaces.MultiDiscrete([self.lateral_bins, self.angular_bins], seed=self.seed)
        else:
            return gym.spaces.Discrete(self.lateral_bins + self.angular_bins, seed=self.seed)

    def _make_observation_space(self) -> spaces.Box:
        """
        Helper to create observation space

        :return: the environment observation space
        """
        if self.image_obs_space:
            return gym.spaces.Box(low=0, high=255, shape=self._get_image_observation().shape, dtype=np.uint8)
        else:
            # In the discrete case, the agent acts on the feature vector representation of the observation
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._get_discrete_observation().shape,
                                  dtype=np.float32)


def get_random_white_striker_position(prob: float):
    # 0.5229
    if prob < 0.25:
        return 0.4479 + 0.075
    elif 0.25 < prob < 0.5:
        return -0.15 + 0.075
    elif 0.5 < prob < 0.75:
        return -0.7456 + 0.075
    else:
        return -1.0435 + 0.075


def calculate_axis_velocities(x_start, y_start, x_target, y_target_range, velocity, random_num_generator):
    """
    Calculates the necessary velocity components (v_x, v_y) to hit a target range.

    Parameters:
    - x_start: Starting x-coordinate (constant)
    - y_start: Starting y-coordinate (randomized)
    - x_target: Target x-coordinate
    - y_target_range: Tuple (y_min, y_max) defining the target range along the y-axis
    - velocity: Magnitude of the velocity
    - random_num_generator: Random number generator

    Returns:
    - Tuple (v_x, v_y): Velocity components along x and y axes.
    """
    y_target = random_num_generator.uniform(low=y_target_range[0], high=y_target_range[1])
    delta_x = x_target - x_start
    delta_y = y_target - y_start

    angle_radians = np.arctan2(delta_y, delta_x)

    v_x = velocity * np.cos(angle_radians)
    v_y = velocity * np.sin(angle_radians)

    return v_x, v_y


def is_ball_outside_goalie_space(ball_data):
    return ball_data[0] > -0.91 or np.abs(ball_data[1]) > 0.29


def is_ball_outside_bounds(ball_data):
    return np.abs(ball_data[0]) > 1.4 or np.abs(ball_data[1]) > 0.71


def is_ball_in_goal_bounds(ball_pos):
    return -0.3 < ball_pos[1] < 0.3 and -0.341 < ball_pos[2] < 0.741


def is_ball_in_black_goal_bounds(ball_pos):
    return -1.376 < ball_pos[0] < -1.216 and is_ball_in_goal_bounds(ball_pos)


def is_ball_in_white_goal_bounds(ball_pos):
    return 1.216 < ball_pos[0] < 1.376 and is_ball_in_goal_bounds(ball_pos)
