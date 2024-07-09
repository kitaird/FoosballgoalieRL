import logging.config
from pathlib import Path

logging.config.fileConfig(Path(__file__).parent / 'logging.ini')
logger = logging.getLogger(__name__)

from eval import evaluate_model
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
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

    logger.info("Starting experiment %s in mode %s on environment %s", experiment_name, experiment_mode, env_id)
    logger.info("Using base directory %s for storing training/testing data and models", base_dir)

    if experiment_mode == 'train':
        train_loop(env_id=env_id, config=config, training_path=base_dir / 'training', algorithm_class=rl_alg)
    elif experiment_mode == 'test':
        evaluate_model(env_id=env_id, config=config, test_path=base_dir / 'testing', algorithm_class=rl_alg)
    else:
        raise ValueError(f"Unknown mode: {experiment_mode}")


if __name__ == '__main__':
    main()
