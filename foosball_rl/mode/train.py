import logging
from pathlib import Path

from foosball_rl.setup.callback_configurator import get_callbacks
from foosball_rl.utils.tensorboard_aggregator import aggregate_results
from stable_baselines3 import HerReplayBuffer  # noqa: F401

from foosball_rl.algorithms.model_loader import get_model
from foosball_rl.create_env import create_envs
from foosball_rl.utils.config import get_run_config
from foosball_rl.utils.utils import log_experiment_config

logger = logging.getLogger(__name__)


def train_loop(env_id: str, algo: str, training_path: Path):
    training_config = get_run_config()['Training']
    for seed in training_config['seeds']:
        logging.info("Creating %s %s envs with seed %s", training_config['n_envs'], env_id, seed)
        env = create_envs(env_id=env_id, n_envs=training_config['n_envs'], seed=seed, video_logging_path=training_path,
                          vec_normalize_path=training_config['vec_normalize_load_path'])
        train(algo=algo, env=env, seed=seed, experiment_path=training_path, training_config=training_config)
    aggregate_results(training_path)

def train(algo: str, env, seed: int,experiment_path: Path, training_config):
    model, used_hyperparameter = get_model(algo=algo, env=env, seed=seed, experiment_path=experiment_path)
    log_experiment_config(hyperparameter=used_hyperparameter, venv=env, save_path=experiment_path, seed=seed)

    tb_log_name = training_config['tb_log_name'] + f'_seed_{seed}'
    model.learn(total_timesteps=training_config['total_timesteps'], tb_log_name=tb_log_name,
                callback=get_callbacks(env, seed, experiment_path))
    env.close()
