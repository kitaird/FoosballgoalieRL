from pathlib import Path

import gymnasium as gym
from configparser import ConfigParser, ExtendedInterpolation

from foosball_rl.environments.foosball.episode_definition import FoosballEpisodeDefinition

env_cfg = ConfigParser(interpolation=ExtendedInterpolation())
env_cfg.read(Path(__file__).parent / 'foosball.cfg')

episode_definition_cfg = env_cfg['EpisodeDefinition']
episode_definition = FoosballEpisodeDefinition()

foosball_id = 'Foosball-v0'

gym.register(
    id=foosball_id,
    entry_point='foosball_rl.environments.foosball.foosball:Foosball',
    max_episode_steps=1000,
    kwargs={
        'step_frequency': env_cfg['Environment'].getint('step_frequency'),
        'render_mode': env_cfg['Environment']['render_mode'],
        'episode_definition': episode_definition,
        'env_config': env_cfg
    }
)

