import logging
from glob import glob
from pathlib import Path
from typing import Dict, Any

import tensorboard_reducer as tbr
from stable_baselines3.common.type_aliases import GymEnv

from foosball_rl import EXPERIMENT_NAME, EXECUTION_MODE, ENV_ID, RL_ALGORITHM
from foosball_rl.callbacks.callback_configurator import CALLBACK_CONFIG
from foosball_rl.wrappers.wrapper_configuration import get_applied_gym_wrappers, get_applied_vecenv_wrappers, \
    ENV_WRAPPERS, VEC_ENV_WRAPPERS

logger = logging.getLogger(__name__)


def aggregate_results(training_path) -> None:
    tensorboard_path = training_path / 'tensorboard'
    reduce_ops = ("mean", "min", "max", "median", "std", "var")
    events_dict = tbr.load_tb_events(sorted(glob(tensorboard_path.__str__() + '/*')))
    n_scalars = len(events_dict)
    n_steps, n_events = list(events_dict.values())[0].shape
    logger.info("Loaded %s TensorBoard runs with %s scalars and %s steps each", n_events, n_scalars, n_steps)
    reduced_events = tbr.reduce_events(events_dict, reduce_ops)
    output_path = tensorboard_path / "aggregates" / "operation"
    for op in reduce_ops:
        logger.debug("Writing \'%s\' reduction to \'%s-%s\'", op, output_path, op)
    tbr.write_tb_events(reduced_events, output_path.__str__(), overwrite=False)


def log_training_config(env: GymEnv, seed: int, save_path: Path, hyperparameter: Dict[str, Any]) -> None:
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / f'{EXPERIMENT_NAME}_training_configuration.txt', 'w') as f:
        f.write('Training Configuration\n')
        f.write('-' * 100 + '\n')
        f.write(f"Experiment name: {EXPERIMENT_NAME}\n")
        f.write(f"Execution mode: {EXECUTION_MODE}\n")
        f.write(f"Environment: {ENV_ID}\n")
        f.write(f"Algorithm: {RL_ALGORITHM}\n")
        f.write(f"Seed: {seed}\n")
        f.write('-' * 100 + '\n')
        f.write('Hyperparameters\n')
        for k, v in hyperparameter.items():
            f.write(f'{k}: {v}\n')
        f.write('-' * 100 + '\n')
        f.write('Applied wrappers\n')
        # f.write(f'Gym Wrappers: {get_applied_gym_wrappers(env.unwrapped.envs[0])}\n')
        f.write(f'VecEnv Wrappers: {get_applied_vecenv_wrappers(env)}\n')
        f.write('-' * 100 + '\n')
        f.write('Environment Arguments\n')
        # env_cfg = env.unwrapped.envs[0].env_config
        # for k, v in env_cfg.items():
        #     f.write(f'{k}: {v}\n')
        # f.write('-' * 100 + '\n')
        # f.write('Env Wrappers\n')
        # for k, v in ENV_WRAPPERS.items():
        #     f.write(f'{k}: {v}\n')
        # f.write('-' * 100 + '\n')
        # f.write('VecEnv Wrappers\n')
        # for k, v in VEC_ENV_WRAPPERS.items():
        #     f.write(f'{k}: {v}\n')
        # f.write('-' * 100 + '\n')
        # f.write('Callbacks\n')
        # for k, v in CALLBACK_CONFIG.items():
        #     f.write(f'{k}: {v}\n')
        # f.write('-' * 100 + '\n')
        # f.write('')


def print_dict(dictionary: Dict[str, Any]) -> None:
    for k, v in dictionary.items():
        print(f'{k}: {v}')