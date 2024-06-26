import ast
import importlib
from configparser import ConfigParser
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from config import save_run_info
from custom_callbacks import TensorboardCallback
from foosball_rl.environment import create_kicker_env

aggregator = importlib.import_module('tensorboard-aggregator')


def train_loop(config: ConfigParser, training_path: Path, algorithm_class):
    for seed in range(1, 4):
        env = create_kicker_env(config=config, seed=seed, experiment_path_for_video_storing=training_path)
        train_kicker(config=config,
                     seed=seed,
                     experiment_path=training_path,
                     env=env,
                     algorithm_class=algorithm_class)
    aggregator.main(path_arg=training_path / 'tensorboard')


def train_kicker(config: ConfigParser, seed: int, experiment_path: Path, algorithm_class, env):
    alg_config = config['Algorithm']
    policy_kwargs = ast.literal_eval(alg_config['policy_kwargs']) if config.has_option('Algorithm', 'policy_kwargs') else None

    model = algorithm_class(env=env, seed=seed, verbose=1,
                            policy=alg_config['policy'],
                            policy_kwargs=policy_kwargs,
                            tensorboard_log=experiment_path / 'tensorboard'
                            ################################
                            # Add here more hyperparameters if needed, following the above scheme
                            # hyperparameter_name=alg_config['hyperparameter_name']
                            ################################
                            )

    save_run_info(config=config,
                  save_path=experiment_path,
                  seed=seed,
                  algorithm_name=type(model).__name__)

    training_config = config['Training']
    model.learn(total_timesteps=int(training_config['total_timesteps']),
                tb_log_name=training_config['tb_log_name'],
                callback=get_callback(config, experiment_path, seed))
    env.close()


def get_callback(config: ConfigParser, experiment_path: Path, seed: int):
    callback_config = config['Callback']
    checkpoint_callback = CheckpointCallback(name_prefix=f"rl_model_seed_{seed}",
                                             save_freq=int(callback_config['save_freq']),
                                             save_path=(experiment_path / 'logs' / 'checkpoints').__str__(),
                                             save_replay_buffer=callback_config.getboolean('save_replay_buffer'),
                                             save_vecnormalize=callback_config.getboolean('save_vecnormalize'))
    logging_callback = TensorboardCallback()
    return CallbackList([checkpoint_callback, logging_callback])
