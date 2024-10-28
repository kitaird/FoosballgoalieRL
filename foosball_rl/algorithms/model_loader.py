import logging
from pathlib import Path
from typing import Dict, Any, Type

import yaml
from sb3_contrib import ARS, QRDQN, TQC, TRPO, RecurrentPPO
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import unwrap_vec_wrapper, VecEnv, VecNormalize

from foosball_rl.environments.common.wrappers.custom_vec_wrappers import VecPBRSWrapper

logger = logging.getLogger(__name__)


ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    # SB3 Contrib,
    "ars": ARS,
    "qrdqn": QRDQN,
    "tqc": TQC,
    "trpo": TRPO,
    "ppo_lstm": RecurrentPPO,
}


def get_model(algo: str, env, seed: int, experiment_path: Path) -> tuple[BaseAlgorithm, Dict[str, Any]]:
    hyperparams = get_hyperparams(algo)

    logger.info("Training with alg %s and hyperparameters: %s", algo, hyperparams)

    # Update discount-factor in relevant wrappers
    update_discount_factor(env, float(hyperparams['gamma']))

    return (ALGOS[algo](env=env, seed=seed, tensorboard_log=(experiment_path / 'tensorboard').__str__(), **hyperparams),
            hyperparams)


def get_hyperparams(algo):
    with open(Path(__file__).parent / 'hyperparams.yml') as f:
        hyperparams_dict = yaml.safe_load(f)
    hyperparams = hyperparams_dict[algo]
    for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
        if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
            hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])
    return hyperparams


def update_discount_factor(venv: VecEnv, discount_factor: float):
    vec_normalize = unwrap_vec_wrapper(venv, VecNormalize)
    if vec_normalize is not None:
        vec_normalize.gamma = discount_factor

    vec_pbrs = unwrap_vec_wrapper(venv, VecPBRSWrapper)
    if vec_pbrs is not None:
        vec_pbrs.gamma = discount_factor
