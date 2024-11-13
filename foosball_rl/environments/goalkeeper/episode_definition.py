import time
from typing import Optional

import numpy as np

from foosball_rl.environments.common.base_episode_definition import EpisodeDefinition
from foosball_rl.environments.common.constants import PLAYER_BALL_DISTANCE_INCREMENT, WHITE_STRIKER_X_POSITION, \
    BLACK_GOAL_X_POSITION, FIELD_HEIGHT, ABS_GOAL_Y_SYMMETRIC_BOUND
from foosball_rl.environments.common.constraints import ball_outside_table, ball_stopped, \
    ball_outside_player_space, ball_in_black_goal_bounds, ball_in_white_goal_bounds


class GoalkeeperEpisodeDefinition(EpisodeDefinition):

    def __init__(self,
                 end_episode_on_struck_goal: bool = True,
                 end_episode_on_conceded_goal: bool = True,
                 reset_goalie_position_on_episode_start: bool = True,
                 end_episode_on_ball_stopped: bool = True,
                 ball_stopped_time_threshold_in_s: float = 5):
        super().__init__()
        self.end_episode_on_struck_goal: bool = end_episode_on_struck_goal
        self.end_episode_on_conceded_goal: bool = end_episode_on_conceded_goal
        self.reset_goalie_position_on_episode_start: bool = reset_goalie_position_on_episode_start
        self.end_episode_on_ball_stopped: bool = end_episode_on_ball_stopped
        self.ball_stopped_time_threshold_in_s: float = ball_stopped_time_threshold_in_s
        self._ball_stopped_since: Optional[float] = None

    def initialize_episode(self):
        if self.reset_goalie_position_on_episode_start:
            qpos = np.zeros(self.mj_data.qpos.shape)
            qvel = np.zeros(self.mj_data.qvel.shape)
        else:
            qpos = self.mj_data.qpos.copy()
            qvel = self.mj_data.qvel.copy()

        ball_x_pos = WHITE_STRIKER_X_POSITION + PLAYER_BALL_DISTANCE_INCREMENT
        ball_y_pos = self.np_random.uniform(low=-0.1, high=0.1)
        ball_z_pos = FIELD_HEIGHT

        ball_x_vel, ball_y_vel = self._calculate_axis_velocities(ball_x_pos,
                                                                 ball_y_pos,
                                                                 BLACK_GOAL_X_POSITION,
                                                                 y_target_range=[-ABS_GOAL_Y_SYMMETRIC_BOUND,
                                                                                 ABS_GOAL_Y_SYMMETRIC_BOUND],
                                                                 velocity=2)
        qpos[0] = ball_x_pos
        qpos[1] = ball_y_pos
        qpos[2] = ball_z_pos
        qvel[0] = ball_x_vel
        qvel[1] = ball_y_vel
        self.mj_data.qpos[:] = qpos
        self.mj_data.qvel[:] = qvel

    def is_truncated(self) -> bool:
        ball_position = self.mj_data.qpos[0:2].copy()
        if ball_outside_table(ball_position):
            return True
        ball_velocity = self.mj_data.qvel[0:2].copy()
        self._ball_stopped_since = time.time() if ball_stopped(ball_velocity) else None
        return self._ball_stopped_since is not None and ball_outside_player_space(ball_position, "b_g") or \
                self.ball_stopped_exceeded_threshold()

    def is_terminated(self) -> bool:
        ball_pos = self.mj_data.qpos[0:3].copy()
        return ((self.end_episode_on_conceded_goal and
                 (self.mj_data.sensordata[0].copy() > 0 or ball_in_black_goal_bounds(ball_pos)))
                or
                (self.end_episode_on_struck_goal and
                 (self.mj_data.sensordata[1].copy() > 0 or ball_in_white_goal_bounds(ball_pos))))

    def ball_stopped_exceeded_threshold(self) -> bool:
        return (self.end_episode_on_ball_stopped and
                self._ball_stopped_since is not None and
                time.time() - self._ball_stopped_since > self.ball_stopped_time_threshold_in_s)

    def _calculate_axis_velocities(self, x_start, y_start, x_target, y_target_range, velocity):
        """
        Calculates the necessary velocity components (v_x, v_y) to hit a target range.

        Parameters:
        - x_start: Starting x-coordinate (constant)
        - y_start: Starting y-coordinate (randomized)
        - x_target: Target x-coordinate
        - y_target_range: Tuple (y_min, y_max) defining the target range along the y-axis
        - velocity: Magnitude of the velocity

        Returns:
        - Tuple (v_x, v_y): Velocity components along x and y axes.
        """
        y_target = self.np_random.uniform(low=y_target_range[0], high=y_target_range[1])
        delta_x = x_target - x_start
        delta_y = y_target - y_start

        angle_radians = np.arctan2(delta_y, delta_x)

        v_x = velocity * np.cos(angle_radians)
        v_y = velocity * np.sin(angle_radians)

        return v_x, v_y

    def __str__(self):
        return f"GoalkeeperEpisodeDefinition(end_episode_on_struck_goal={self.end_episode_on_struck_goal}, " \
               f"end_episode_on_conceded_goal={self.end_episode_on_conceded_goal}, " \
               f"reset_goalie_position_on_episode_start={self.reset_goalie_position_on_episode_start})"
