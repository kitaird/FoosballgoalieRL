from pathlib import Path

from stable_baselines3 import PPO, SAC

from eval import evaluate_model
from foosball_rl.config.config import get_config
from foosball_rl.environments.foosball import foosball_id
from foosball_rl.environments.goalkeeper import goalkeeper_id
from train import train_loop

possible_envs = [foosball_id, goalkeeper_id]  # Import for triggering gym registration


def main():
    config = get_config()
    experiment_name = config['Common']['experiment_name']
    experiment_mode = config['Common']['mode']
    env_id = config['Common']['env_id']
    base_dir = Path(__file__).resolve().parent.parent / 'experiments' / experiment_name

    rl_alg = globals()[config['Algorithm']['algo']]

    print("-" * 50)
    print(f"Starting experiment {experiment_name} in mode {experiment_mode} on environment {env_id}")
    print(f"Using base directory {base_dir} for storing training/testing data and models")
    print("-" * 50)

    if experiment_mode == 'train':
        train_loop(env_id=env_id, config=config, training_path=base_dir / 'training', algorithm_class=rl_alg)
    elif experiment_mode == 'test':
        evaluate_model(env_id=env_id, config=config, test_path=base_dir / 'testing', algorithm_class=rl_alg)
    else:
        raise ValueError(f"Unknown mode: {experiment_mode}")


if __name__ == '__main__':
    main()
