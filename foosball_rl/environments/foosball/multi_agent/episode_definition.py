import time
from typing import Optional

import numpy as np

from foosball_rl.environments.common.base_episode_definition import EpisodeDefinition
from foosball_rl.environments.common.constants import FIELD_HEIGHT
from foosball_rl.environments.common.constraints import ball_outside_table, ball_in_black_goal_bounds, \
    ball_in_white_goal_bounds, ball_stopped


class FoosballEpisodeDefinition(EpisodeDefinition):

    def __init__(self):
        super().__init__()
        self.ball_stopped_time_threshold_in_s: float = 1
        self._ball_stopped_since: Optional[float] = None

    def initialize_episode(self):
        qpos = np.zeros(self.mj_data.qpos.shape)
        qvel = np.zeros(self.mj_data.qvel.shape)

        ball_x_pos = self.np_random.uniform(low=-0.1, high=0.1)
        ball_y_pos = self.np_random.uniform(low=-0.2, high=0.2)
        ball_z_pos = FIELD_HEIGHT + 0.01
        ball_x_vel = self.np_random.uniform(low=-0.1, high=0.1)
        ball_y_vel = self.np_random.uniform(low=-0.1, high=0.1)

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
        if ball_stopped(ball_velocity):
            if self._ball_stopped_since is None:
                self._ball_stopped_since = time.time()
        else:
            self._ball_stopped_since = None
        return self._ball_stopped_since is not None and self.ball_stopped_exceeded_threshold()

    def is_terminated(self) -> bool:
        ball_pos = self.mj_data.qpos[0:3].copy()
        return (self.mj_data.sensordata[0].copy() > 0 or ball_in_black_goal_bounds(ball_pos)) or (
                    self.mj_data.sensordata[1].copy() > 0 or ball_in_white_goal_bounds(ball_pos))

    def ball_stopped_exceeded_threshold(self) -> bool:
        return time.time() - self._ball_stopped_since > self.ball_stopped_time_threshold_in_s
