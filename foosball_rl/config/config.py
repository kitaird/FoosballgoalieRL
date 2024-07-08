from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path


config_path = base_dir = Path(__file__).resolve().parent / 'run_config.ini'


def get_config():
    if not hasattr(get_config, 'config'):
        get_config.config = ConfigParser(interpolation=ExtendedInterpolation())
        get_config.config.read(config_path)
    return get_config.config


LINE_SEPARATOR = '-----------------\n'


def save_run_info(run_config: ConfigParser, env_config: ConfigParser, save_path: Path, seed: int, algorithm_name: str):
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / 'run_configuration.txt', 'w') as f:
        f.write('Run Configuration\n')
        f.write(LINE_SEPARATOR)
        f.write(f"Experiment name: {run_config['Common']['experiment_name']}\n")
        f.write(f"Environment: {run_config['Common']['env_id']}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(LINE_SEPARATOR)
        f.write('Environment Arguments\n')
        print_cfg(f, env_config)
        f.write(LINE_SEPARATOR)
        f.write('Run Arguments\n')
        print_cfg(f, run_config)
        f.write(LINE_SEPARATOR)
        f.write('')


def print_cfg(f, cfg: ConfigParser):
    for k, v in cfg._sections.items():
        if isinstance(v, dict):
            f.write(LINE_SEPARATOR)
            f.write(f'{k}\n')
            for i in v.items():
                f.write(f'\t{i}\n')
            f.write(LINE_SEPARATOR)