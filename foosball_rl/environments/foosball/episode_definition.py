import numpy as np

from foosball_rl.environments.base_episode_definition import EpisodeDefinition
from foosball_rl.environments.constraints import ball_outside_table, black_goal_scored, white_goal_scored


class FoosballEpisodeDefinition(EpisodeDefinition):

    def __init__(self):
        super().__init__()

    def initialize_episode(self):
        self.mj_data.qpos[:] = np.zeros(self.mj_data.qpos.shape)
        self.mj_data.qvel[:] = np.zeros(self.mj_data.qvel.shape)

    def is_truncated(self) -> bool:
        return ball_outside_table(self.mj_data.body("ball").xpos)

    def is_terminated(self) -> bool:
        ball_pos = self.mj_data.body("ball").xpos
        return black_goal_scored(self.mj_data.sensor, ball_pos) or white_goal_scored(self.mj_data.sensor, ball_pos)
