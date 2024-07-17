from pathlib import Path

import gymnasium as gym
import yaml

from foosball_rl.environments.goalkeeper.episode_definition import GoalkeeperEpisodeDefinition

with open(Path(__file__).parent / 'goalkeeper-config.yml') as f:
    env_cfg = yaml.safe_load(f)

episode_definition_cfg = env_cfg['EpisodeDefinition']
episode_definition = GoalkeeperEpisodeDefinition(
    reset_goalie_position_on_episode_start=episode_definition_cfg['reset_goalie_position_on_episode_start'],
    end_episode_on_struck_goal=episode_definition_cfg['end_episode_on_struck_goal'],
    end_episode_on_conceded_goal=episode_definition_cfg['end_episode_on_conceded_goal'])

goalkeeper_id = 'Goalkeeper-v0'

gym.register(
    id=goalkeeper_id,
    entry_point='foosball_rl.environments.goalkeeper.goalkeeper:RawEnv',
    max_episode_steps=1000,
    kwargs={
        'step_frequency': env_cfg['Environment']['step_frequency'],
        'render_mode': env_cfg['Environment']['render_mode'],
        'use_image_obs': env_cfg['Environment']['use_image_obs'],
        'episode_definition': episode_definition,
        'env_config': env_cfg
    }
)
