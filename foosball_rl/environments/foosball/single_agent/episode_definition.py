import numpy as np

from foosball_rl.environments.common.base_episode_definition import EpisodeDefinition
from foosball_rl.environments.common.constants import FIELD_HEIGHT
from foosball_rl.environments.common.constraints import ball_outside_table, ball_in_black_goal_bounds, \
    ball_in_white_goal_bounds


class FoosballEpisodeDefinition(EpisodeDefinition):

    def __init__(self):
        super().__init__()

    def initialize_episode(self):
        qpos = np.zeros(self.mj_data.qpos.shape)
        qvel = np.zeros(self.mj_data.qvel.shape)

        ball_x_pos = 0.0
        ball_y_pos = 0.0
        ball_z_pos = FIELD_HEIGHT
        ball_x_vel = self.np_random.uniform(low=-0.005, high=0.005)
        ball_y_vel = self.np_random.uniform(low=-0.05, high=0.05)

        qpos[0] = ball_x_pos
        qpos[1] = ball_y_pos
        qpos[2] = ball_z_pos
        qvel[0] = ball_x_vel
        qvel[1] = ball_y_vel
        self.mj_data.qpos[:] = qpos
        self.mj_data.qvel[:] = qvel

    def is_truncated(self) -> bool:
        return ball_outside_table(self.mj_data.qpos[0:3].copy())

    def is_terminated(self) -> bool:
        ball_pos = self.mj_data.qpos[0:3].copy()
        return (self.mj_data.sensordata[0].copy() > 0 or ball_in_black_goal_bounds(ball_pos)) or (
                    self.mj_data.sensordata[1].copy() > 0 or ball_in_white_goal_bounds(ball_pos))
