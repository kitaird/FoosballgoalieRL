import logging
from pathlib import Path

import yaml
from stable_baselines3 import HerReplayBuffer  # noqa: F401
from stable_baselines3.common.type_aliases import GymEnv

from foosball_rl.algorithms.model_loader import get_model
from foosball_rl.callbacks.callback_configurator import get_callbacks
from foosball_rl.environments.create_env import create_envs, create_marl_envs
from foosball_rl.logging.logging_utils import aggregate_results, log_training_config

logger = logging.getLogger(__name__)


def train_loop(env_id: str, algo: str, training_path: Path) -> None:
    config_path = Path(__file__).parent / 'execution_mode_config.yml'
    with open(config_path) as f:
        training_config = yaml.safe_load(f)['Training']

    for seed in training_config['seeds']:
        logging.info("Creating %s %s envs with seed %s", training_config['n_envs'], env_id, seed)
        env = create_marl_envs(use_team_agents=True, n_envs=training_config['n_envs'], seed=seed,
                               video_logging_path=training_path,
                               vec_normalize_path=training_config['vec_normalize_load_path'])
        # env = create_envs(env_id=env_id, n_envs=training_config['n_envs'], seed=seed, video_logging_path=training_path,
        #                   vec_normalize_path=training_config['vec_normalize_load_path'])
        train(algo=algo, env=env, seed=seed, experiment_path=training_path, training_config=training_config)
    aggregate_results(training_path)


def train(algo: str, env: GymEnv, seed: int, experiment_path: Path, training_config) -> None:
    model, used_hyperparameter = get_model(algo=algo, env=env, seed=seed, experiment_path=experiment_path)
    log_training_config(env=env, seed=seed, save_path=experiment_path, hyperparameter=used_hyperparameter)

    tb_log_name = training_config['tb_log_name'] + f'_seed_{seed}'
    model.learn(total_timesteps=training_config['total_timesteps'], tb_log_name=tb_log_name,
                callback=get_callbacks(env, seed, experiment_path))
    env.close()
