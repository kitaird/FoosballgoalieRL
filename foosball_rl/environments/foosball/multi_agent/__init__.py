from pathlib import Path

import gymnasium as gym
import yaml

from foosball_rl.environments.foosball.multi_agent.episode_definition import FoosballEpisodeDefinition
from foosball_rl.environments.foosball.single_agent.foosball import RawEnv

with open(Path(__file__).parent / 'foosball-config.yml') as f:
    env_cfg = yaml.safe_load(f)

episode_definition_cfg = env_cfg['EpisodeDefinition']
episode_definition = FoosballEpisodeDefinition()

foosball2v2_id = 'Foosball-2v2-v0'
foosball4v4_id = 'Foosball-4v4-v0'

env_creator = lambda: RawEnv(step_frequency=env_cfg['Environment']['step_frequency'],
        render_mode=env_cfg['Environment']['render_mode'],
        use_image_obs=env_cfg['Environment']['use_image_obs'],
        episode_definition=episode_definition,
        env_config=env_cfg)

gym.register(
    id=foosball2v2_id,
    entry_point='foosball_rl.environments.foosball.multi_agent.foosball_marl:FoosballMARL',
    max_episode_steps=env_cfg['Environment']['horizon'],
    kwargs={
        'env_creator': env_creator,
        'use_team_agents': True
    }
)

gym.register(
    id=foosball4v4_id,
    entry_point='foosball_rl.environments.foosball.multi_agent.foosball_marl:FoosballMARL',
    max_episode_steps=env_cfg['Environment']['horizon'],
    kwargs={
        'env_creator': env_creator,
        'use_team_agents': False
    }
)

