import time
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path
from typing import Dict, Any

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from foosball_rl.create_env import create_env

logged_callback_values = defaultdict(list)


def evaluate_model(env_id: str, config: ConfigParser, test_path: Path, algorithm_class):
    test_cfg = config['Testing']

    print("-" * 50)
    print(f"Evaluating {algorithm_class.__name__} model from {test_cfg['test_model_path']} on {env_id} environment")

    model_path = test_cfg['model_path']
    model = algorithm_class.load(model_path)

    env = create_env(env_id=env_id, config=config, seed=test_cfg.getint('eval_seed'), video_logging_path=test_path)
    if test_cfg['normalized_env_path'] is not None:
        print(f"Using normalized environment from {test_cfg['normalized_env_path']}")
        env = VecNormalize.load(venv=env, load_path=test_cfg['normalized_env_path'])

    episode_rewards, episode_lengths = evaluate_policy(model=model, env=env,
                                                       n_eval_episodes=test_cfg.getint('num_eval_episodes'),
                                                       callback=_log_callback)

    save_results(config=config, test_path=test_path, model_path=model_path, episode_rewards=episode_rewards,
                 episode_lengths=episode_lengths, callback_values=logged_callback_values)

    print(f"Mean reward: {episode_rewards}, Mean episode length: {episode_lengths}")
    print("-" * 50)


def save_results(config: ConfigParser, test_path: Path, model_path: str, episode_rewards: float, episode_lengths: float,
                 callback_values: Dict[str, Any] = None):
    eval_file_name = f'evaluation_result_{model_path[model_path.rindex("/")+1:]}_{round(time.time() * 1000)}.txt'
    with open(test_path / eval_file_name, 'w') as f:
        f.write(f"Experiment name: {config['Common']['experiment_name']}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Evaluation seed: {config['Testing'].getint('eval_seed')}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Number of evaluation episodes: {config['Testing'].getint('num_eval_episodes')}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean reward: {episode_rewards}\n")
        f.write(f"Mean episode length: {episode_lengths}\n")
        f.write("-" * 50 + "\n")
        f.write("Callback values:\n")
        for k, v in callback_values.items():
            f.write(f"{k}: {v}\n")
        f.write("-" * 50 + "\n")


def _log_callback(locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
    """
    :param locals_:
    :param globals_:
    """
    info = locals_["info"]

    ball_position = info["ball_position"]
    logged_callback_values["custom/ball_position_x"].append(ball_position[0])
    logged_callback_values["custom/ball_position_y"].append(ball_position[1])
    logged_callback_values["custom/ball_position_z"].append(ball_position[2])
