from pathlib import Path
from configparser import ConfigParser

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, VecVideoRecorder)
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from custom_wrappers import VecPBRSWrapper
from kicker.kicker_env import Kicker


def create_kicker_env(config: ConfigParser, seed: int, experiment_path_for_video_storing: Path):
    env_conf = config['Kicker']
    env = Kicker(seed=seed,
                 horizon=int(env_conf['horizon']),
                 continuous_act_space=env_conf.getboolean('continuous_act_space'),
                 multi_discrete_act_space=env_conf.getboolean('multi_discrete_act_space'),
                 image_obs_space=env_conf.getboolean('image_obs_space'),
                 end_episode_on_struck_goal=env_conf.getboolean('end_episode_on_struck_goal'),
                 end_episode_on_conceded_goal=env_conf.getboolean('end_episode_on_conceded_goal'),
                 reset_goalie_position=env_conf.getboolean('reset_goalie_position'),
                 render_training=env_conf.getboolean('render_training'),
                 lateral_bins=env_conf.getint('lateral_bins'),
                 angular_bins=env_conf.getint('angular_bins'),
                 step_frequency=env_conf.getint('step_frequency'))
    # Default wrappers
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    ############################################
    # Add Wrappers here
    ############################################
    # env = ThisIsAnExampleWrapper(env)
    env = VecPBRSWrapper(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    ############################################
    if not env_conf.getboolean('render_training'):
        video_log_path = experiment_path_for_video_storing / 'logs' / 'video'
        video_conf = config['VideoRecording']
        print(f"Recording video every {video_conf.getint('video_interval')} steps with a length of "
              f"{video_conf.getint('video_length')} frames, saving to {video_log_path}")
        env = VecVideoRecorder(venv=env, name_prefix=f"rl-kicker-video-seed-{seed}",
                               record_video_trigger=lambda x: x % video_conf.getint('video_interval') == 0,
                               video_length=video_conf.getint('video_length'),
                               video_folder=video_log_path)
    env.seed(seed)
    return env


def load_normalized_kicker_env(config: ConfigParser, seed: int, experiment_path: Path, normalize_path: str):
    env = create_kicker_env(config=config, seed=seed, experiment_path_for_video_storing=experiment_path)
    env = VecNormalize.load(normalize_path, env)
    return env
