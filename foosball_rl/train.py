import ast
import importlib
import json
from configparser import ConfigParser
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from foosball_rl.common.custom_callbacks import TensorboardCallback
from foosball_rl.config.config import save_run_info
from foosball_rl.create_env import create_env

tensorboard_aggregator = importlib.import_module('tensorboard-aggregator.aggregator')


def train_loop(env_id: str, config: ConfigParser, training_path: Path, algorithm_class):
    for seed in json.loads(config['Training']['seeds']):
        env = create_env(env_id=env_id, config=config, seed=seed, video_logging_path=training_path)
        train(config=config, seed=seed, env=env, experiment_path=training_path, algorithm_class=algorithm_class)
    tensorboard_aggregator.main(path_arg=training_path / 'tensorboard')


def train(config: ConfigParser, seed: int, experiment_path: Path, algorithm_class, env):
    alg_config = config['Algorithm']
    policy_kwargs = ast.literal_eval(alg_config['policy_kwargs']) if config.has_option('Algorithm', 'policy_kwargs') else None

    model = algorithm_class(env=env, seed=seed, verbose=1,
                            policy=alg_config['policy'],
                            gamma=alg_config.getfloat('discount_factor'),
                            policy_kwargs=policy_kwargs,
                            tensorboard_log=experiment_path / 'tensorboard'
                            ################################
                            # Add here more hyperparameters if needed, following the above scheme
                            # hyperparameter_name=alg_config['hyperparameter_name']
                            # Make sure to pass the correct type, as parsing might not always work as expected
                            ################################
                            )

    save_run_info(run_config=config, env_config=env.unwrapped.get_attr('env_config')[0], save_path=experiment_path,
                  seed=seed, algorithm_name=type(model).__name__)

    training_config = config['Training']
    model.learn(total_timesteps=training_config.getint('total_timesteps'), tb_log_name=training_config['tb_log_name'],
                callback=get_callbacks(config, experiment_path, seed))
    env.close()


def get_callbacks(config: ConfigParser, experiment_path: Path, seed: int):
    callback_config = config['Callbacks']
    checkpoint_callback = CheckpointCallback(name_prefix=f"rl_model_seed_{seed}",
                                             save_freq=int(callback_config['save_freq']),
                                             save_path=(experiment_path / 'checkpoints').__str__(),
                                             save_replay_buffer=callback_config.getboolean('save_replay_buffer'),
                                             save_vecnormalize=callback_config.getboolean('save_vecnormalize'))
    logging_callback = TensorboardCallback()
    ############################################
    # Add more callbacks here if needed
    ############################################
    return CallbackList([checkpoint_callback, logging_callback])
