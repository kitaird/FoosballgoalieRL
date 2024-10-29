import logging.config
import time
from pathlib import Path

import yaml

from foosball_rl import EXPERIMENT_NAME, EXECUTION_MODE, ENV_ID, RL_ALGORITHM

with open(Path(__file__).parent / 'logging' / 'logging_config.yml') as f:
    log_cfg = yaml.safe_load(f)
    logging.config.dictConfig(log_cfg)

import foosball_rl.environments  # noqa: F401
from foosball_rl.modes.train import train_loop
from foosball_rl.modes.eval import evaluate_model

logger = logging.getLogger(__name__)


def main():
    base_dir = Path(__file__).parent.parent / 'experiments' / EXPERIMENT_NAME

    logger.info("Starting experiment %s with execution mode %s on environment %s", EXPERIMENT_NAME, EXECUTION_MODE, ENV_ID)
    logger.info("Using base directory %s for storing training/testing data and models", base_dir)

    if EXECUTION_MODE == 'train':
        training_path = base_dir / 'training'
        if training_path.exists():
            training_path = rewrite_path_if_exists(training_path)
        train_loop(env_id=ENV_ID, algo=RL_ALGORITHM, training_path=training_path)
    elif EXECUTION_MODE == 'eval':
        evaluation_path = base_dir / 'evaluation'
        if evaluation_path.exists():
            evaluation_path = rewrite_path_if_exists(evaluation_path)
        evaluate_model(env_id=ENV_ID, algo=RL_ALGORITHM, eval_path=evaluation_path)
    else:
        raise ValueError(f"Unknown execution mode: {EXECUTION_MODE}")


def rewrite_path_if_exists(path: Path):
    logger.warning("File or directory %s already exists, appending with current timestamp", path)
    return path.with_name(path.name + '_' + str(round(time.time() * 1000)))


if __name__ == '__main__':
    main()
