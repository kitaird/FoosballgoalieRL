from pathlib import Path

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, ProgressBarCallback, CallbackList

from foosball_rl.create_env import create_eval_envs
from foosball_rl.environments.common.custom_callbacks import TensorboardCallback, \
    SaveVecNormalizeAndRolloutBufferCallback
from foosball_rl.utils.config import get_run_config


def get_callbacks(venv, seed: int, experiment_path: Path):
    callback_config = get_run_config()['Callbacks']
    callbacks = [TensorboardCallback()]

    if callback_config['use_eval_callback']:
        eval_callback = get_eval_callback(callback_config, experiment_path, seed, venv)
        callbacks.append(eval_callback)

    if callback_config['use_checkpoint_callback']:
        get_checkpoint_callback(callback_config, callbacks, experiment_path, seed)

    if callback_config['use_progress_bar_callback']:
        callbacks.append(ProgressBarCallback())
    ############################################
    # Add more callbacks here if needed
    ############################################
    return CallbackList(callbacks)


def get_eval_callback(callback_config, experiment_path, seed, venv):
    eval_callback_config = callback_config['EvalCallback']
    eval_path = experiment_path / f'seed-{seed}' / 'eval'
    eval_callback = EvalCallback(
        eval_env=create_eval_envs(env_id=venv.unwrapped.envs[0].spec.id,
                                  n_envs=eval_callback_config['eval_n_envs'],
                                  seed=eval_callback_config['eval_seed'],
                                  video_logging_path=eval_path / 'video'),
        callback_on_new_best=SaveVecNormalizeAndRolloutBufferCallback(save_freq=1, save_path=eval_path / 'best'),
        best_model_save_path=(eval_path / 'best').__str__(),
        n_eval_episodes=eval_callback_config['eval_n_episodes'],
        log_path=(eval_path / 'log').__str__(),
        eval_freq=eval_callback_config['eval_freq'],
        deterministic=eval_callback_config['eval_deterministic'],
    )
    return eval_callback


def get_checkpoint_callback(callback_config, callbacks, experiment_path, seed):
    checkpoint_callback_config = callback_config['CheckpointCallback']
    callbacks.append(CheckpointCallback(
        name_prefix="rl_model",
        save_freq=int(checkpoint_callback_config['checkpoint_save_freq']),
        save_path=(experiment_path / f'seed-{seed}' / 'checkpoints').__str__(),
        save_replay_buffer=checkpoint_callback_config['checkpoint_save_replay_buffer'],
        save_vecnormalize=checkpoint_callback_config['checkpoint_save_vecnormalize']))
