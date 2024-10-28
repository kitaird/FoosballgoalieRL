from pathlib import Path

import gymnasium as gym
import yaml

from foosball_rl.environments.foosball.single_agent.episode_definition import FoosballEpisodeDefinition

with open(Path(__file__).parent / 'foosball-config.yml') as f:
    env_cfg = yaml.safe_load(f)

episode_definition_cfg = env_cfg['EpisodeDefinition']
episode_definition = FoosballEpisodeDefinition()

foosball_id = 'Foosball-v0'

gym.register(
    id=foosball_id,
    entry_point='foosball_rl.environments.foosball.single_agent.foosball_rl:RawEnv',
    max_episode_steps=env_cfg['Environment']['horizon'],
    kwargs={
        'step_frequency': env_cfg['Environment']['step_frequency'],
        'render_mode': env_cfg['Environment']['render_mode'],
        'episode_definition': episode_definition,
        'env_config': env_cfg
    }
)

