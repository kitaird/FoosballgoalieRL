import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import yaml
from stable_baselines3.common.evaluation import evaluate_policy

from foosball_rl import EXPERIMENT_NAME
from foosball_rl.algorithms.model_loader import ALGOS
from foosball_rl.environments.create_env import create_eval_envs

logger = logging.getLogger(__name__)

logged_callback_values = defaultdict(list)


def evaluate_model(env_id: str, algo: str, eval_path: Path) -> None:
    config_path = Path(__file__).parent / 'execution_mode_config.yml'
    with open(config_path) as f:
        eval_config = yaml.safe_load(f)['Evaluation']

    model_path = eval_config['model_path']
    eval_seed = eval_config['eval_seed']
    n_eval_episodes = eval_config['n_eval_episodes']

    logger.info("Evaluating Alg: %s loaded from %s on %s environment", algo, model_path, env_id)

    model = ALGOS[algo].load(model_path)

    venv = create_eval_envs(env_id, n_eval_envs=n_eval_episodes, seed=eval_seed, video_logging_path=eval_path,
                            vec_normalize_path=eval_config['vec_normalize_load_path'])

    episode_rewards, episode_lengths = evaluate_policy(model=model, env=venv, n_eval_episodes=n_eval_episodes,
                                                       callback=_log_callback)

    save_results(eval_path=eval_path,
                 eval_seed=eval_seed,
                 n_eval_episodes=n_eval_episodes,
                 model_path=model_path,
                 episode_rewards=episode_rewards,
                 episode_lengths=episode_lengths,
                 callback_values=logged_callback_values)

    logger.info("Mean reward: %s, Mean episode length: %s", episode_rewards, episode_lengths)


def save_results(eval_path: Path,
                 model_path: str,
                 eval_seed: int,
                 n_eval_episodes: int,
                 episode_rewards: float,
                 episode_lengths: float,
                 callback_values: Dict[str, Any] = None) -> None:
    eval_file_name = f'evaluation_result_{EXPERIMENT_NAME}_{model_path[model_path.rindex("/") + 1:]}_{round(time.time() * 1000)}.txt'
    with open(eval_path / eval_file_name, 'w') as f:
        f.write(f"Experiment name: {EXPERIMENT_NAME}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Evaluation seed: {eval_seed}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Number of evaluation episodes: {n_eval_episodes}\n")
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
    ##############################
    # <<ExtensionPoint>>: You can add custom callback logging here
    ##############################
    # info = locals_["info"]
    # ball_position = info["ball_position"]
    # logged_callback_values["custom/ball_position_x"].append(ball_position[0])
    # logged_callback_values["custom/ball_position_y"].append(ball_position[1])
    # logged_callback_values["custom/ball_position_z"].append(ball_position[2])
