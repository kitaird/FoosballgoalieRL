import yaml
from pathlib import Path
from typing import Dict, Any

from foosball_rl.misc.utils import get_applied_gym_wrappers, get_applied_vecenv_wrappers

config_path = base_dir = Path(__file__).resolve().parent.parent / 'run_config.yml'


def get_run_config():
    if not hasattr(get_run_config, 'config'):
        with open(config_path) as f:
            get_run_config.config = yaml.safe_load(f)
    return get_run_config.config


LINE_SEPARATOR = '-----------------\n'


def save_run_info(hyperparams: Dict[str, Any], venv, save_path: Path, seed: int):
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / 'run_configuration.txt', 'w') as f:
        f.write('Run Configuration\n')
        f.write(LINE_SEPARATOR)
        f.write(f"Experiment name: {get_run_config()['Common']['experiment_name']}\n")
        f.write(f"Environment: {get_run_config()['Common']['env_id']}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Algorithm: {get_run_config()['Common']['algorithm']}\n")
        f.write(LINE_SEPARATOR)
        f.write('Hyperparameters\n')
        for k, v in hyperparams.items():
            f.write(f'{k}: {v}\n')
        f.write(LINE_SEPARATOR)
        f.write('Applied wrappers\n')
        f.write(f'Gym Wrappers: {get_applied_gym_wrappers(venv.unwrapped.envs[0])}\n')
        f.write(f'VecEnv Wrappers: {get_applied_vecenv_wrappers(venv)}\n')
        f.write(LINE_SEPARATOR)
        f.write('Environment Arguments\n')
        env_cfg = venv.unwrapped.envs[0].env_config
        for k, v in env_cfg.items():
            f.write(f'{k}\n')
            for i in v.items():
                f.write(f'\t{i}\n')
        f.write(LINE_SEPARATOR)
        f.write('Run Arguments\n')
        for k, v in get_run_config().items():
            f.write(f'{k}\n')
            for i in v.items():
                f.write(f'\t{i}\n')
        # print_cfg(f, run_config)
        f.write(LINE_SEPARATOR)
        f.write('')
