from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, ProgressBarCallback, CallbackList

from foosball_rl.environments.create_env import create_eval_envs
from foosball_rl.environments.common.custom_callbacks import TensorboardCallback, \
    SaveVecNormalizeAndRolloutBufferCallback

callback_config_path = Path(__file__).parent / 'callback_config.yml'

with open(callback_config_path) as f:
    callback_conf = yaml.safe_load(f)

CALLBACK_CONFIG = callback_conf['Callbacks']


def get_callbacks(venv, seed: int, experiment_path: Path):
    callbacks = []

    if CALLBACK_CONFIG['use_tensorboard_callback']:
        callbacks.append(TensorboardCallback())

    if CALLBACK_CONFIG['use_eval_callback']:
        eval_callback = get_eval_callback(experiment_path, seed, venv)
        callbacks.append(eval_callback)

    if CALLBACK_CONFIG['use_checkpoint_callback']:
        get_checkpoint_callback(callbacks, experiment_path, seed)

    if CALLBACK_CONFIG['use_progress_bar_callback']:
        callbacks.append(ProgressBarCallback())
    ############################################
    # <<ExtensionPoint>>: Add more callbacks here if needed
    ############################################
    return CallbackList(callbacks)


def get_eval_callback(experiment_path, seed, venv):
    eval_callback_config = CALLBACK_CONFIG['EvalCallback']
    eval_path = experiment_path / f'seed-{seed}' / 'eval'
    eval_callback = EvalCallback(
        eval_env=create_eval_envs(env_id=venv.unwrapped.envs[0].spec.id,
                                  n_eval_envs=eval_callback_config['n_eval_envs'],
                                  seed=eval_callback_config['eval_seed'],
                                  video_logging_path=eval_path / 'video'),
        callback_on_new_best=SaveVecNormalizeAndRolloutBufferCallback(save_freq=1, save_path=eval_path / 'best'),
        best_model_save_path=(eval_path / 'best').__str__(),
        n_eval_episodes=eval_callback_config['n_eval_episodes'],
        log_path=(eval_path / 'log').__str__(),
        eval_freq=eval_callback_config['eval_freq'],
        deterministic=eval_callback_config['eval_deterministic'],
    )
    return eval_callback


def get_checkpoint_callback(callbacks, experiment_path, seed):
    checkpoint_callback_config = CALLBACK_CONFIG['CheckpointCallback']
    callbacks.append(CheckpointCallback(
        name_prefix=checkpoint_callback_config['name_prefix'],
        save_freq=int(checkpoint_callback_config['save_freq']),
        save_path=(experiment_path / f'seed-{seed}' / 'checkpoints').__str__(),
        save_replay_buffer=checkpoint_callback_config['save_replay_buffer'],
        save_vecnormalize=checkpoint_callback_config['save_vecnormalize']))
