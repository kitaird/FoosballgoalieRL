import logging.config
import time
from pathlib import Path

import yaml

with open(Path(__file__).parent / 'utils' / 'logging.yml') as f:
    log_cfg = yaml.safe_load(f)
    logging.config.dictConfig(log_cfg)

import foosball_rl.environments  # noqa: F401
from foosball_rl.utils.config import get_run_config
from foosball_rl.train import train_loop
from foosball_rl.eval import evaluate_model

logger = logging.getLogger(__name__)


def main():
    config = get_run_config()
    experiment_name = config['Common']['experiment_name']
    experiment_mode = config['Common']['mode']
    env_id = config['Common']['env_id']
    base_dir = Path(__file__).resolve().parent.parent / 'experiments' / experiment_name

    rl_algo = config['Common']['algorithm']

    logger.info("Starting experiment %s in mode %s on environment %s", experiment_name, experiment_mode, env_id)
    logger.info("Using base directory %s for storing training/testing data and models", base_dir)

    if experiment_mode == 'train':
        training_path = base_dir / 'training'
        if training_path.exists():
            training_path = rewrite_path_if_exists(training_path)
        train_loop(env_id=env_id, algo=rl_algo, training_path=training_path)
    elif experiment_mode == 'test':
        testing_path = base_dir / 'testing'
        if testing_path.exists():
            testing_path = rewrite_path_if_exists(testing_path)
        evaluate_model(env_id=env_id, algo=rl_algo, test_path=testing_path)
    else:
        raise ValueError(f"Unknown mode: {experiment_mode}")


def rewrite_path_if_exists(path: Path):
    logger.warning("File or directory %s already exists, appending with current timestamp", path)
    return path.with_name(path.name + '_' + str(round(time.time() * 1000)))


if __name__ == '__main__':
    main()
