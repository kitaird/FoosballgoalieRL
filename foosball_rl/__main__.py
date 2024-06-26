from pathlib import Path

from stable_baselines3 import PPO

from config import get_config
from evaluate import evaluate_model
from train import train_loop


def main():
    config = get_config()
    experiment_mode = config['Common']['mode']
    experiment_name = config['Common']['experiment_name']
    base_dir = Path(__file__).resolve().parent.parent / 'experiments' / experiment_name

    if experiment_mode == 'train':
        training_path = base_dir / 'training'
        train_loop(config=config, training_path=training_path, algorithm_class=PPO)
    elif experiment_mode == 'test':
        test_path = base_dir / 'testing'
        evaluate_model(config=config, test_path=test_path, algorithm_class=PPO)
    else:
        raise ValueError(f"Unknown mode: {experiment_mode}")


if __name__ == '__main__':
    main()
