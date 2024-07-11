import numpy as np

from foosball_rl.environments.base_episode_definition import EpisodeDefinition
from foosball_rl.environments.constants import FIELD_HEIGHT
from foosball_rl.environments.constraints import ball_outside_table, black_goal_scored, white_goal_scored


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
        return ball_outside_table(self.mj_data.body("ball").xpos)

    def is_terminated(self) -> bool:
        ball_pos = self.mj_data.body("ball").xpos
        return black_goal_scored(self.mj_data.sensor, ball_pos) or white_goal_scored(self.mj_data.sensor, ball_pos)
