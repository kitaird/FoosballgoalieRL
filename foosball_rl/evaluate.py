from configparser import ConfigParser
from pathlib import Path
from environment import load_normalized_kicker_env
from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_model(config: ConfigParser, test_path: Path, algorithm_class):
    test_cfg = config['Testing']
    model = algorithm_class.load(test_cfg['test_model_path'])
    env = load_normalized_kicker_env(config=config, seed=test_cfg.getint('eval_seed'), experiment_path=test_path,
                                     normalize_path=test_cfg['normalized_env_path'])
    episode_rewards, episode_lengths = evaluate_policy(model=model, env=env,
                                                       n_eval_episodes=test_cfg.getint('num_eval_episodes'))
    save_results(config=config, test_cfg=test_cfg, test_path=test_path,
                 episode_rewards=episode_rewards, episode_lengths=episode_lengths)

    print("-" * 50)
    print(f"Mean reward: {episode_rewards}, Mean episode length: {episode_lengths}")
    print("-" * 50)


def save_results(config: ConfigParser, test_cfg, test_path: Path, episode_rewards, episode_lengths):
    with open(test_path / 'evaluation_result.txt', 'w') as f:
        f.write(f"Experiment name: {config['Common']['experiment_name']}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Seed: {test_cfg.getint('eval_seed')}\n")
        f.write(f"Number of evaluation episodes: {test_cfg.getint('num_eval_episodes')}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean reward: {episode_rewards}\n")
        f.write(f"Mean episode length: {episode_lengths}\n")
        f.write("-" * 50 + "\n")