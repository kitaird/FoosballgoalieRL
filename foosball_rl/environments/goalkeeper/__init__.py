from pathlib import Path

import gymnasium as gym
from configparser import ConfigParser, ExtendedInterpolation

from foosball_rl.environments.goalkeeper.episode_definition import GoalkeeperEpisodeDefinition

env_cfg = ConfigParser(interpolation=ExtendedInterpolation())
env_cfg.read(Path(__file__).parent / 'goalkeeper.cfg')

episode_definition_cfg = env_cfg['EpisodeDefinition']
episode_definition = GoalkeeperEpisodeDefinition(
    reset_goalie_position_on_episode_start=episode_definition_cfg.getboolean('reset_goalie_position_on_episode_start'),
    end_episode_on_struck_goal=episode_definition_cfg.getboolean('end_episode_on_struck_goal'),
    end_episode_on_conceded_goal=episode_definition_cfg.getboolean('end_episode_on_conceded_goal'))

goalkeeper_id = 'Goalkeeper-v0'

gym.register(
    id=goalkeeper_id,
    entry_point='foosball_rl.environments.goalkeeper.goalkeeper:Goalkeeper',
    max_episode_steps=1000,
    kwargs={
        'step_frequency': env_cfg['Environment'].getint('step_frequency'),
        'render_mode': env_cfg['Environment']['render_mode'],
        'use_image_obs': env_cfg['Environment'].getboolean('use_image_obs'),
        'episode_definition': episode_definition,
        'env_config': env_cfg
    }
)
