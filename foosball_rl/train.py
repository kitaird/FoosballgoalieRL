import logging
import tensorboard_reducer as tbr
from glob import glob
from pathlib import Path
from typing import Dict, Any

import yaml
from stable_baselines3 import HerReplayBuffer  # noqa: F401
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, ProgressBarCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, unwrap_vec_wrapper

from foosball_rl.misc.config import save_run_info, get_run_config
from foosball_rl.create_env import create_envs, create_eval_envs
from foosball_rl.environments.common.custom_callbacks import TensorboardCallback, \
    SaveVecNormalizeAndRolloutBufferCallback
from foosball_rl.environments.common.custom_vec_wrappers import VecPBRSWrapper
from foosball_rl.misc.utils import ALGOS

logger = logging.getLogger(__name__)


def train_loop(env_id: str, algo: str, training_path: Path):
    training_config = get_run_config()['Training']
    for seed in training_config['seeds']:
        logging.info("Creating %s %s envs with seed %s", training_config['n_envs'], env_id, seed)
        env = create_envs(env_id=env_id, n_envs=training_config['n_envs'], seed=seed,
                          video_logging_path=training_path, vec_normalize_path=training_config['vec_normalize_load_path'])
        train(algo=algo, env=env, seed=seed, experiment_path=training_path)
    aggregate_results(training_path)


def aggregate_results(training_path):
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


def train(algo: str, env, seed: int, experiment_path: Path):
    model, used_hyperparams = get_model(algo=algo, env=env, seed=seed, experiment_path=experiment_path)

    save_run_info(hyperparams=used_hyperparams, venv=env, save_path=experiment_path, seed=seed)

    training_config = get_run_config()['Training']
    tb_log_name = training_config['tb_log_name'] + f'_seed_{seed}'
    model.learn(total_timesteps=training_config['total_timesteps'], tb_log_name=tb_log_name,
                callback=get_callbacks(env, experiment_path, seed))
    env.close()


def get_model(algo: str, env, seed: int, experiment_path: Path) -> tuple[BaseAlgorithm, Dict[str, Any]]:
    with open(Path(__file__).parent / 'hyperparams.yml') as f:
        hyperparams_dict = yaml.safe_load(f)

    hyperparams = hyperparams_dict[algo]
    for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
        if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
            hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

    logger.info("Training with alg %s and hyperparameters: %s", algo, hyperparams)

    # Update discount-factor in relevant wrappers
    unwrap_vec_wrapper(env, VecNormalize).gamma = float(hyperparams['gamma'])
    unwrap_vec_wrapper(env, VecPBRSWrapper).gamma = float(hyperparams['gamma'])

    return (ALGOS[algo](env=env, seed=seed, tensorboard_log=(experiment_path / 'tensorboard').__str__(), **hyperparams),
            hyperparams)


def get_callbacks(env, experiment_path: Path, seed: int):
    callback_config = get_run_config()['Callbacks']

    eval_path = experiment_path / f'seed-{seed}' / 'eval'
    eval_callback = EvalCallback(
        eval_env=create_eval_envs(env_id=env.unwrapped.get_attr('spec')[0].id,
                                  n_envs=callback_config['eval_n_envs'],
                                  seed=callback_config['eval_seed'],
                                  video_logging_path=eval_path / 'video'),
        callback_on_new_best=SaveVecNormalizeAndRolloutBufferCallback(save_freq=1, save_path=eval_path / 'best'),
        best_model_save_path=(eval_path / 'best').__str__(),
        n_eval_episodes=callback_config['eval_n_episodes'],
        log_path=(eval_path / 'log').__str__(),
        eval_freq=callback_config['eval_freq'],
        deterministic=callback_config['eval_deterministic'],
    )
    callbacks = [eval_callback, TensorboardCallback()]

    if callback_config['use_checkpoint_callback']:
        callbacks.append(CheckpointCallback(
            name_prefix=f"rl_model",
            save_freq=int(callback_config['checkpoint_save_freq']),
            save_path=(experiment_path / f'seed-{seed}' / 'checkpoints').__str__(),
            save_replay_buffer=callback_config['checkpoint_save_replay_buffer'],
            save_vecnormalize=callback_config['checkpoint_save_vecnormalize']))

    if callback_config['show_progress_bar']:
        callbacks.append(ProgressBarCallback())
    ############################################
    # Add more callbacks here if needed
    ############################################
    return CallbackList(callbacks)
