import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

from stable_baselines3.common.evaluation import evaluate_policy

from foosball_rl.misc.config import get_run_config
from foosball_rl.create_env import create_eval_envs
from foosball_rl.misc.utils import ALGOS

logger = logging.getLogger(__name__)

logged_callback_values = defaultdict(list)


def evaluate_model(env_id: str, algo: str, test_path: Path):
    test_cfg = get_run_config()['Testing']

    model_path = test_cfg['model_path']
    logger.info("Evaluating Alg: %s loaded from %s on %s environment", algo, model_path, env_id)

    model = ALGOS[algo].load(model_path)

    venv = create_eval_envs(env_id, n_envs=test_cfg['n_envs'], seed=test_cfg['eval_seed'],
                            video_logging_path=test_path, vec_normalize_path=test_cfg['vec_normalize_load_path'])

    episode_rewards, episode_lengths = evaluate_policy(model=model, env=venv,
                                                       n_eval_episodes=test_cfg['num_eval_episodes'],
                                                       callback=_log_callback)

    save_results(test_path=test_path, model_path=model_path, episode_rewards=episode_rewards,
                 episode_lengths=episode_lengths, callback_values=logged_callback_values)

    logger.info("Mean reward: %s, Mean episode length: %s", episode_rewards, episode_lengths)


def save_results(test_path: Path, model_path: str, episode_rewards: float,
                 episode_lengths: float,
                 callback_values: Dict[str, Any] = None):
    eval_file_name = f'evaluation_result_{model_path[model_path.rindex("/") + 1:]}_{round(time.time() * 1000)}.txt'
    with open(test_path / eval_file_name, 'w') as f:
        f.write(f"Experiment name: {get_run_config()['Common']['experiment_name']}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Evaluation seed: {get_run_config()['Testing']['eval_seed']}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Number of evaluation episodes: {get_run_config()['Testing']['num_eval_episodes']}\n")
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
    # Custom callback logging
    ##############################
    # info = locals_["info"]
    # ball_position = info["ball_position"]
    # logged_callback_values["custom/ball_position_x"].append(ball_position[0])
    # logged_callback_values["custom/ball_position_y"].append(ball_position[1])
    # logged_callback_values["custom/ball_position_z"].append(ball_position[2])
